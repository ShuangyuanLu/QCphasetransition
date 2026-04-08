from abc import ABC, abstractmethod

import numpy as np

from binary_spin_symmetry import (
    CompiledBinarySpinKernel,
    get_binary_spin_symmetry_data,
    rectangular_support_offsets,
)
from mc_analysis import analyze_measurements


class PeriodicState:
    def __init__(self, n_sample, lattice_size, dtype=np.int8):
        self.n_sample = n_sample
        self.lattice_size = lattice_size
        self.sites = np.zeros((n_sample, lattice_size + 1, lattice_size + 1), dtype=dtype)

    def bulk_view(self):
        return self.sites[:, 0:self.lattice_size, 0:self.lattice_size]

    def sync_periodic_boundaries(self, row=None, column=None):
        # The last row/column are ghost cells. Keep them synchronized with the
        # wrapped physical edge so vectorized slices see periodic boundaries.
        if row == 0:
            self.sites[:, 0, :] = self.sites[:, self.lattice_size, :]
        elif row == 1:
            self.sites[:, self.lattice_size, :] = self.sites[:, 0, :]

        if column == 0:
            self.sites[:, :, 0] = self.sites[:, :, self.lattice_size]
        elif column == 1:
            self.sites[:, :, self.lattice_size] = self.sites[:, :, 0]


class TranslationallySymmetricUpdate(ABC):
    def __init__(self, name, unit_cell_shape, support_offsets, shift=(0, 0)):
        self.name = name
        self.unit_cell_shape = unit_cell_shape
        self.support_offsets = tuple(support_offsets)
        self.shift = shift

    @abstractmethod
    def apply(self, state, rng):
        pass


class TiledUpdate(TranslationallySymmetricUpdate):
    def __init__(self, name, unit_cell_shape, support_offsets, shift=(0, 0), boundary_sync=(None, None)):
        super().__init__(name, unit_cell_shape, support_offsets, shift)
        self.boundary_sync = boundary_sync

        for row_offset, column_offset in self.support_offsets:
            if row_offset < 0 or row_offset >= self.unit_cell_shape[0]:
                raise ValueError("support row offset must lie inside the unit cell")
            if column_offset < 0 or column_offset >= self.unit_cell_shape[1]:
                raise ValueError("support column offset must lie inside the unit cell")

    def tile_views(self, state):
        # Each support offset selects one site inside the unit cell. Stepping by
        # the unit-cell size walks over every translated copy of that local site.
        views = {}
        row_step, column_step = self.unit_cell_shape
        shift_row, shift_column = self.shift
        for row_offset, column_offset in self.support_offsets:
            row_start = shift_row + row_offset
            column_start = shift_column + column_offset
            views[(row_offset, column_offset)] = state.sites[
                :,
                row_start:row_start + state.lattice_size:row_step,
                column_start:column_start + state.lattice_size:column_step,
            ]
        return views

    def apply(self, state, rng):
        tile_views = self.tile_views(state)
        new_tiles = self.transform_tiles(tile_views, rng)
        for offset, tile_view in tile_views.items():
            tile_view[...] = new_tiles[offset]
        state.sync_periodic_boundaries(*self.boundary_sync)

    @abstractmethod
    def transform_tiles(self, tile_views, rng):
        pass


class BinarySpinLocalRule(ABC):
    @abstractmethod
    def transform(self, tile_views, rng):
        pass


class BinarySpinTiledUpdate(TiledUpdate):
    def __init__(
        self,
        name,
        unit_cell_shape,
        support_offsets,
        shift=(0, 0),
        boundary_sync=(None, None),
        local_rule=None,
    ):
        super().__init__(name, unit_cell_shape, support_offsets, shift, boundary_sync)
        if local_rule is None:
            raise ValueError("BinarySpinTiledUpdate requires a local_rule")
        self.local_rule = local_rule

    def transform_tiles(self, tile_views, rng):
        updated_tiles = {offset: tile.copy() for offset, tile in tile_views.items()}
        for offset, updated_tile in self.local_rule.transform(tile_views, rng).items():
            updated_tiles[offset] = np.asarray(updated_tile, dtype=tile_views[offset].dtype)
        return updated_tiles


def rectangular_boundary_sync(support_shape, shift):
    row_starts = {shift[0] + row for row in range(support_shape[0])}
    column_starts = {shift[1] + column for column in range(support_shape[1])}

    row_sync = None
    if 0 in row_starts:
        row_sync = 1
    elif support_shape[0] in row_starts:
        row_sync = 0

    column_sync = None
    if 0 in column_starts:
        column_sync = 1
    elif support_shape[1] in column_starts:
        column_sync = 0

    return row_sync, column_sync


class Observer(ABC):
    def __init__(self):
        self.measurement = {}

    @abstractmethod
    def observe(self, state):
        pass

    def record(self, state):
        values = self.observe(state)
        for key, value in values.items():
            self.measurement.setdefault(key, []).append(value)
        return values


class IsingMagnetizationObserver(Observer):
    def observe(self, state):
        bulk = state.bulk_view()
        lattice_area = state.lattice_size * state.lattice_size
        magnetization = 1 - 2 * np.sum(bulk, axis=(1, 2)) / lattice_area
        return {
            "m": np.sum(magnetization) / state.n_sample,
            "m_2": np.sum(np.square(magnetization)) / state.n_sample,
            "m_4": np.sum(np.power(magnetization, 4)) / state.n_sample,
        }


class PeriodicUpdateRunner:
    def __init__(self, state, updates, observer, n_periods, measure_every_periods, random_seed=0):
        if n_periods <= 0:
            raise ValueError("n_periods must be positive")
        if measure_every_periods <= 0:
            raise ValueError("measure_every_periods must be positive")

        self.state = state
        self.updates = list(updates)
        self.observer = observer
        self.n_periods = n_periods
        self.measure_every_periods = measure_every_periods
        self.random_seed = random_seed

    def run(self):
        rng = np.random.RandomState(self.random_seed)
        for period_index in range(self.n_periods):
            for update in self.updates:
                update.apply(self.state, rng)
            if period_index % self.measure_every_periods == 0:
                self.observer.record(self.state)
        return self.observer.measurement


class SingleSiteFlipRule(BinarySpinLocalRule):
    def __init__(self, p_flip, offset=(0, 0)):
        self.p_flip = p_flip
        self.offset = offset

    def transform(self, tile_views, rng):
        tile = tile_views[self.offset]
        flip_mask = rng.rand(*tile.shape) < self.p_flip
        return {
            self.offset: np.logical_xor(tile, flip_mask).astype(tile.dtype, copy=False),
        }


class PairAlignmentRule(BinarySpinLocalRule):
    def __init__(self, p_align, offsets):
        if len(offsets) != 2:
            raise ValueError("PairAlignmentRule requires exactly two support offsets")
        self.p_align = p_align
        self.offsets = tuple(offsets)

    def transform(self, tile_views, rng):
        offset_a, offset_b = self.offsets
        tile_a = tile_views[offset_a]
        tile_b = tile_views[offset_b]

        update_mask = (rng.rand(*tile_a.shape) < self.p_align) & np.logical_xor(tile_a, tile_b)
        aligned_value = (rng.rand(*tile_a.shape) < 0.5).astype(tile_a.dtype, copy=False)

        new_tile_a = np.where(update_mask, aligned_value, tile_a)
        new_tile_b = np.where(update_mask, aligned_value, tile_b)
        return {
            offset_a: new_tile_a.astype(tile_a.dtype, copy=False),
            offset_b: new_tile_b.astype(tile_b.dtype, copy=False),
        }


class TwoByTwoPlaquetteRule(BinarySpinLocalRule):
    def transform(self, tile_views, rng):
        tile_00 = tile_views[(0, 0)]
        tile_01 = tile_views[(0, 1)]
        tile_10 = tile_views[(1, 0)]
        tile_11 = tile_views[(1, 1)]

        block_mask = rng.rand(*tile_00.shape) < 0.5
        cluster = tile_00 + tile_10 + tile_01 + tile_11

        new_tiles = {}
        for offset, tile in tile_views.items():
            updated_tile = np.logical_or(cluster == 3, tile)
            updated_tile = np.logical_and(np.logical_not(cluster == 1), updated_tile)
            updated_tile = np.logical_xor(cluster == 2, updated_tile)
            new_tiles[offset] = np.where(block_mask, updated_tile, tile).astype(tile.dtype, copy=False)
        return new_tiles


class SymmetricKernelRule(BinarySpinLocalRule):
    def __init__(self, support_shape, orbit_logits):
        self.kernel = CompiledBinarySpinKernel.from_orbit_logits(support_shape, orbit_logits)
        self.support_offsets = self.kernel.symmetry_data.support_offsets

    def transform(self, tile_views, rng):
        input_state_indices = self.kernel.encode_tile_views(tile_views, self.support_offsets)
        output_state_indices = self.kernel.sample_next_states(input_state_indices, rng)
        output_bits = self.kernel.decode_states(output_state_indices)

        updated_tiles = {}
        for bit_index, offset in enumerate(self.support_offsets):
            updated_tiles[offset] = output_bits[..., bit_index].astype(tile_views[offset].dtype, copy=False)
        return updated_tiles


class SingleSiteFlipUpdate(BinarySpinTiledUpdate):
    def __init__(self, name, p_flip, shift=(0, 0)):
        if tuple(shift) != (0, 0):
            raise ValueError("single_site_flip only supports shift=(0, 0) because the unit cell is (1, 1)")
        super().__init__(
            name=name,
            unit_cell_shape=(1, 1),
            support_offsets=((0, 0),),
            shift=(0, 0),
            boundary_sync=(1, 1),
            local_rule=SingleSiteFlipRule(p_flip),
        )


class IsingOnsiteFlipUpdate(SingleSiteFlipUpdate):
    def __init__(self, p_flip):
        super().__init__(name="onsite_flip", p_flip=p_flip, shift=(0, 0))


class PairAlignmentRectangularUpdate(BinarySpinTiledUpdate):
    def __init__(self, name, support_shape, shift, p_align):
        support_shape = tuple(support_shape)
        support_offsets = rectangular_support_offsets(support_shape)
        if len(support_offsets) != 2:
            raise ValueError("pair_alignment requires exactly two sites, so support_shape must be (1, 2) or (2, 1)")

        super().__init__(
            name=name,
            unit_cell_shape=support_shape,
            support_offsets=support_offsets,
            shift=shift,
            boundary_sync=rectangular_boundary_sync(support_shape, shift),
            local_rule=PairAlignmentRule(p_align, support_offsets),
        )


class HorizontalPairAlignmentUpdate(BinarySpinTiledUpdate):
    def __init__(self, shift_column, p_align):
        super().__init__(
            name=f"horizontal_pair_align_{shift_column}",
            unit_cell_shape=(1, 2),
            support_offsets=((0, 0), (0, 1)),
            shift=(0, shift_column),
            boundary_sync=(1, 1 - shift_column),
            local_rule=PairAlignmentRule(p_align, ((0, 0), (0, 1))),
        )


class VerticalPairAlignmentUpdate(BinarySpinTiledUpdate):
    def __init__(self, shift_row, p_align):
        support_offsets = ((0, 0), (1, 0))
        super().__init__(
            name=f"vertical_pair_align_{shift_row}",
            unit_cell_shape=(2, 1),
            support_offsets=support_offsets,
            shift=(shift_row, 0),
            boundary_sync=(1 - shift_row, 1),
            local_rule=PairAlignmentRule(p_align, support_offsets),
        )


class PlaquetteUpdate(BinarySpinTiledUpdate):
    def __init__(self, name, shift):
        super().__init__(
            name=name,
            unit_cell_shape=(2, 2),
            support_offsets=((0, 0), (0, 1), (1, 0), (1, 1)),
            shift=shift,
            boundary_sync=(1 - shift[0], 1 - shift[1]),
            local_rule=TwoByTwoPlaquetteRule(),
        )


class IsingBlockUpdate(PlaquetteUpdate):
    def __init__(self, shift):
        super().__init__(name=f"block_update_{shift[0]}{shift[1]}", shift=shift)


class SymmetricBinarySpinRectangularUpdate(BinarySpinTiledUpdate):
    def __init__(self, name, support_shape, shift, orbit_logits):
        support_offsets = rectangular_support_offsets(support_shape)
        super().__init__(
            name=name,
            unit_cell_shape=support_shape,
            support_offsets=support_offsets,
            shift=shift,
            boundary_sync=rectangular_boundary_sync(support_shape, shift),
            local_rule=SymmetricKernelRule(support_shape, orbit_logits),
        )


def _as_pair(value, field_name):
    value = tuple(value)
    if len(value) != 2:
        raise ValueError(f"{field_name} must have length 2, got {value}")
    return value


def _normalize_shift(shift):
    return _as_pair(shift, "shift")


def _normalize_support_shape(support_shape):
    support_shape = _as_pair(support_shape, "support_shape")
    if support_shape[0] <= 0 or support_shape[1] <= 0:
        raise ValueError(f"support_shape must be positive, got {support_shape}")
    return support_shape


def _normalize_update_shifts(update_spec):
    has_shift = "shift" in update_spec
    has_shifts = "shifts" in update_spec
    if has_shift and has_shifts:
        raise ValueError("update spec can define either shift or shifts, not both")

    if has_shifts:
        raw_shifts = update_spec["shifts"]
    elif has_shift:
        raw_shifts = [update_spec["shift"]]
    else:
        raw_shifts = [(0, 0)]

    return [_normalize_shift(shift) for shift in raw_shifts]


def _expand_update_name(base_name, family, shift, n_shifts):
    if base_name is None:
        return f"{family}_{shift[0]}{shift[1]}"
    if n_shifts == 1:
        return base_name
    return f"{base_name}_{shift[0]}{shift[1]}"


def to_json_safe(value):
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _theta_from_params(params):
    has_theta = "theta" in params
    has_orbit_logits = "orbit_logits" in params
    if has_theta and has_orbit_logits:
        raise ValueError("symmetric_kernel params can define either theta or orbit_logits, not both")
    if has_theta:
        return np.asarray(params["theta"], dtype=np.float64)
    if has_orbit_logits:
        return np.asarray(params["orbit_logits"], dtype=np.float64)
    raise KeyError("symmetric_kernel requires params['theta'] or params['orbit_logits']")


def build_update_from_spec(update_spec, shift=None, update_name=None):
    family = update_spec["family"]
    params = update_spec.get("params", {})

    if shift is None:
        shifts = _normalize_update_shifts(update_spec)
        if len(shifts) != 1:
            raise ValueError("build_update_from_spec expects exactly one shift; use build_period_from_spec for shift lists")
        shift = shifts[0]
    else:
        shift = _normalize_shift(shift)

    if family == "single_site_flip":
        if "p_flip" not in params:
            raise KeyError("single_site_flip requires params['p_flip']")
        return SingleSiteFlipUpdate(update_name or "single_site_flip", params["p_flip"], shift=shift)

    if family == "pair_alignment":
        if "p_align" not in params:
            raise KeyError("pair_alignment requires params['p_align']")
        support_shape = _normalize_support_shape(update_spec["support_shape"])
        if support_shape not in ((1, 2), (2, 1)):
            raise ValueError(f"pair_alignment only supports support_shape (1, 2) or (2, 1), got {support_shape}")
        return PairAlignmentRectangularUpdate(update_name or "pair_alignment", support_shape, shift, params["p_align"])

    if family in {"plaquette", "ising_block"}:
        support_shape = tuple(update_spec.get("support_shape", (2, 2)))
        if support_shape != (2, 2):
            raise ValueError(f"{family} requires support_shape (2, 2), got {support_shape}")
        return PlaquetteUpdate(update_name or "plaquette", shift)

    if family == "symmetric_kernel":
        support_shape = _normalize_support_shape(update_spec["support_shape"])
        expected_orbits = n_transition_orbits_for_shape(support_shape)
        theta = _theta_from_params(params)
        if theta.shape != (expected_orbits,):
            raise ValueError(
                f"symmetric_kernel for support_shape {support_shape} expects theta of shape "
                f"({expected_orbits},), got {theta.shape}"
            )
        return SymmetricBinarySpinRectangularUpdate(update_name or "symmetric_kernel", support_shape, shift, theta)

    raise ValueError(f"unknown update family: {family}")


def build_period_from_spec(period_spec):
    updates = []
    for update_spec in period_spec:
        shifts = _normalize_update_shifts(update_spec)
        base_name = update_spec.get("name")
        for shift in shifts:
            update_name = _expand_update_name(base_name, update_spec["family"], shift, len(shifts))
            updates.append(build_update_from_spec(update_spec, shift=shift, update_name=update_name))
    return updates


def build_observer_from_spec(observer_spec):
    if observer_spec is None:
        return IsingMagnetizationObserver()

    kind = observer_spec.get("kind", "ising_magnetization")
    if kind == "ising_magnetization":
        return IsingMagnetizationObserver()
    raise ValueError(f"unknown observer kind: {kind}")


def build_runner_from_spec(spec):
    simulation_parameters = spec["simulation"]
    period_spec = spec["period"]
    observer = build_observer_from_spec(spec.get("observer"))
    updates = build_period_from_spec(period_spec)
    return build_binary_spin_runner(simulation_parameters, updates, observer=observer)


def make_symmetric_kernel_update_spec(name, support_shape, shifts=None, theta=None):
    support_shape = _normalize_support_shape(support_shape)
    if shifts is None:
        shifts = [(0, 0)]
    shifts = [_normalize_shift(shift) for shift in shifts]

    expected_orbits = n_transition_orbits_for_shape(support_shape)
    if theta is None:
        theta = [0.0] * expected_orbits
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (expected_orbits,):
        raise ValueError(
            f"symmetric_kernel for support_shape {support_shape} expects theta of shape "
            f"({expected_orbits},), got {theta.shape}"
        )

    return to_json_safe({
        "name": name,
        "family": "symmetric_kernel",
        "support_shape": support_shape,
        "shifts": shifts,
        # Keep theta JSON-safe for cross-project use.
        "params": {"theta": theta.tolist()},
    })


def make_model_spec(simulation_parameters, period_spec, observer_kind="ising_magnetization", analysis_spec=None):
    spec = {
        "simulation": dict(simulation_parameters),
        "observer": {"kind": observer_kind},
        "period": list(period_spec),
    }
    if analysis_spec is not None:
        spec["analysis"] = dict(analysis_spec)
    return to_json_safe(spec)


def _measurement_length(measurements):
    if not measurements:
        return 0
    first_key = next(iter(measurements))
    return len(measurements[first_key])


def summarize_measurements(measurements, tail_start_fraction=0.5):
    if not 0.0 <= tail_start_fraction < 1.0:
        raise ValueError("tail_start_fraction must lie in [0, 1)")

    summary = {}
    for key, values in measurements.items():
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            summary[key] = {"n": 0, "mean": None, "tail_mean": None, "last": None}
            continue

        tail_start = int(array.size * tail_start_fraction)
        summary[key] = {
            "n": int(array.size),
            "mean": float(np.mean(array)),
            "tail_mean": float(np.mean(array[tail_start:])),
            "last": float(array[-1]),
        }
    return summary


def _compiled_update_metadata(update):
    return {
        "name": update.name,
        "unit_cell_shape": list(update.unit_cell_shape),
        "support_offsets": [list(offset) for offset in update.support_offsets],
        "shift": list(update.shift),
    }


def evaluate_spec(spec, include_measurements=True, include_final_state=False, tail_start_fraction=0.5):
    runner = build_runner_from_spec(spec)
    measurements = runner.run()
    simulation_parameters = spec["simulation"]
    measurement_stride = simulation_parameters.get("measure_every_periods", simulation_parameters.get("T_measure"))

    result = {
        "n_compiled_updates": len(runner.updates),
        "compiled_updates": [_compiled_update_metadata(update) for update in runner.updates],
        "n_measurements": _measurement_length(measurements),
        "measurement_stride_periods": int(measurement_stride),
        "summary": summarize_measurements(measurements, tail_start_fraction=tail_start_fraction),
        "analysis": analyze_measurements(
            measurements,
            measurement_stride=measurement_stride,
            analysis_spec=spec.get("analysis"),
        ),
    }

    if include_measurements:
        result["measurements"] = {
            key: np.asarray(values, dtype=np.float64).tolist() for key, values in measurements.items()
        }

    if include_final_state:
        result["final_state"] = runner.state.bulk_view().astype(int).tolist()

    return to_json_safe(result)


def build_ising_period(p_flip):
    return [
        IsingOnsiteFlipUpdate(p_flip),
        IsingBlockUpdate((0, 0)),
        IsingBlockUpdate((1, 0)),
        IsingBlockUpdate((1, 1)),
        IsingBlockUpdate((0, 1)),
    ]


def build_pair_period(p_align):
    return [
        HorizontalPairAlignmentUpdate(0, p_align),
        HorizontalPairAlignmentUpdate(1, p_align),
        VerticalPairAlignmentUpdate(0, p_align),
        VerticalPairAlignmentUpdate(1, p_align),
    ]


def build_onsite_and_pair_period(p_flip, p_align):
    return [
        IsingOnsiteFlipUpdate(p_flip),
        *build_pair_period(p_align),
    ]


def build_symmetric_pair_period(pair_orbit_logits):
    return [
        SymmetricBinarySpinRectangularUpdate("symmetric_horizontal_pair_0", (1, 2), (0, 0), pair_orbit_logits),
        SymmetricBinarySpinRectangularUpdate("symmetric_horizontal_pair_1", (1, 2), (0, 1), pair_orbit_logits),
        SymmetricBinarySpinRectangularUpdate("symmetric_vertical_pair_0", (2, 1), (0, 0), pair_orbit_logits),
        SymmetricBinarySpinRectangularUpdate("symmetric_vertical_pair_1", (2, 1), (1, 0), pair_orbit_logits),
    ]


def build_onsite_and_symmetric_pair_period(p_flip, pair_orbit_logits):
    return [
        IsingOnsiteFlipUpdate(p_flip),
        *build_symmetric_pair_period(pair_orbit_logits),
    ]


def n_transition_orbits_for_shape(support_shape):
    return get_binary_spin_symmetry_data(tuple(support_shape)).n_transition_orbits


def build_binary_spin_runner(parameters, updates, observer=None):
    n_periods = parameters.get("n_periods", parameters.get("T"))
    measure_every_periods = parameters.get("measure_every_periods", parameters.get("T_measure"))
    if n_periods is None:
        raise KeyError("parameters must define n_periods or T")
    if measure_every_periods is None:
        raise KeyError("parameters must define measure_every_periods or T_measure")

    if observer is None:
        observer = IsingMagnetizationObserver()

    state = PeriodicState(parameters["N_sample"], parameters["L"])
    return PeriodicUpdateRunner(
        state=state,
        updates=updates,
        observer=observer,
        n_periods=n_periods,
        measure_every_periods=measure_every_periods,
        random_seed=parameters["random_seed"],
    )


def build_ising_runner(parameters, hamiltonian_parameters):
    return build_binary_spin_runner(
        parameters=parameters,
        updates=build_ising_period(hamiltonian_parameters["p_flip"]),
        observer=IsingMagnetizationObserver(),
    )
