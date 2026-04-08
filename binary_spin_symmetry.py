from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def rectangular_support_offsets(support_shape):
    rows, columns = support_shape
    return tuple((row, column) for row in range(rows) for column in range(columns))


def encode_binary_bits(bit_array):
    bit_array = np.asarray(bit_array, dtype=np.int64)
    bit_weights = 1 << np.arange(bit_array.shape[-1], dtype=np.int64)
    return np.tensordot(bit_array, bit_weights, axes=([-1], [0]))


def decode_state_index(state_index, n_bits):
    state_index = np.asarray(state_index, dtype=np.int64)
    return ((state_index[..., None] >> np.arange(n_bits, dtype=np.int64)) & 1).astype(np.int8)


def _rectangle_spatial_permutations(support_shape):
    rows, columns = support_shape
    coordinates = rectangular_support_offsets(support_shape)
    coordinate_to_index = {coordinate: index for index, coordinate in enumerate(coordinates)}

    transforms = [
        lambda row, column: (row, column),
        lambda row, column: (rows - 1 - row, columns - 1 - column),
        lambda row, column: (row, columns - 1 - column),
        lambda row, column: (rows - 1 - row, column),
    ]
    if rows == columns:
        transforms.extend(
            [
                lambda row, column: (column, rows - 1 - row),
                lambda row, column: (columns - 1 - column, row),
                lambda row, column: (column, row),
                lambda row, column: (columns - 1 - column, rows - 1 - row),
            ]
        )

    permutations = []
    seen = set()
    for transform in transforms:
        permutation = []
        for row, column in coordinates:
            transformed_coordinate = transform(row, column)
            permutation.append(coordinate_to_index[transformed_coordinate])
        permutation = tuple(permutation)
        if permutation not in seen:
            seen.add(permutation)
            permutations.append(permutation)
    return tuple(permutations)


@dataclass(frozen=True)
class BinarySpinSymmetryData:
    support_shape: tuple[int, int]
    support_offsets: tuple[tuple[int, int], ...]
    state_bits: np.ndarray
    spatial_permutations: tuple[tuple[int, ...], ...]
    state_actions: np.ndarray
    transition_orbit_ids: np.ndarray
    orbit_representatives: tuple[tuple[int, int], ...]
    orbit_sizes: tuple[int, ...]

    @property
    def n_sites(self):
        return len(self.support_offsets)

    @property
    def n_states(self):
        return self.state_bits.shape[0]

    @property
    def n_transition_orbits(self):
        return len(self.orbit_representatives)

    def format_state(self, state_index):
        bits = self.state_bits[state_index]
        rows, columns = self.support_shape
        return tuple(tuple(int(bit) for bit in bits[row * columns:(row + 1) * columns]) for row in range(rows))

    def orbit_summary(self):
        summary = []
        for orbit_id, (input_state, output_state) in enumerate(self.orbit_representatives):
            summary.append(
                {
                    "orbit_id": orbit_id,
                    "size": self.orbit_sizes[orbit_id],
                    "input_state": self.format_state(input_state),
                    "output_state": self.format_state(output_state),
                }
            )
        return summary


@lru_cache(maxsize=None)
def get_binary_spin_symmetry_data(support_shape):
    support_shape = tuple(support_shape)
    support_offsets = rectangular_support_offsets(support_shape)
    n_sites = len(support_offsets)
    n_states = 1 << n_sites

    state_bits = decode_state_index(np.arange(n_states, dtype=np.int64), n_sites)
    spatial_permutations = _rectangle_spatial_permutations(support_shape)

    state_actions = []
    for permutation in spatial_permutations:
        permuted_bits = state_bits[:, permutation]
        state_actions.append(encode_binary_bits(permuted_bits))
        state_actions.append(encode_binary_bits(1 - permuted_bits))
    state_actions = np.array(state_actions, dtype=np.int64)

    transition_orbit_ids = -np.ones((n_states, n_states), dtype=np.int64)
    orbit_representatives = []
    orbit_sizes = []

    for input_state in range(n_states):
        for output_state in range(n_states):
            if transition_orbit_ids[input_state, output_state] != -1:
                continue

            orbit_id = len(orbit_representatives)
            orbit_representatives.append((input_state, output_state))
            stack = [(input_state, output_state)]
            transition_orbit_ids[input_state, output_state] = orbit_id
            orbit_size = 0

            while stack:
                source_state, target_state = stack.pop()
                orbit_size += 1
                for state_action in state_actions:
                    transformed_source = int(state_action[source_state])
                    transformed_target = int(state_action[target_state])
                    if transition_orbit_ids[transformed_source, transformed_target] == -1:
                        transition_orbit_ids[transformed_source, transformed_target] = orbit_id
                        stack.append((transformed_source, transformed_target))

            orbit_sizes.append(orbit_size)

    return BinarySpinSymmetryData(
        support_shape=support_shape,
        support_offsets=support_offsets,
        state_bits=state_bits,
        spatial_permutations=spatial_permutations,
        state_actions=state_actions,
        transition_orbit_ids=transition_orbit_ids,
        orbit_representatives=tuple(orbit_representatives),
        orbit_sizes=tuple(orbit_sizes),
    )


class CompiledBinarySpinKernel:
    def __init__(self, symmetry_data, orbit_logits):
        self.symmetry_data = symmetry_data
        self.orbit_logits = np.asarray(orbit_logits, dtype=np.float64)
        if self.orbit_logits.shape != (self.symmetry_data.n_transition_orbits,):
            raise ValueError(
                "orbit_logits must have shape "
                f"({self.symmetry_data.n_transition_orbits},), got {self.orbit_logits.shape}"
            )

        transition_logits = self.orbit_logits[self.symmetry_data.transition_orbit_ids]
        transition_logits = transition_logits - np.max(transition_logits, axis=1, keepdims=True)
        transition_weights = np.exp(transition_logits)
        self.transition_matrix = transition_weights / np.sum(transition_weights, axis=1, keepdims=True)
        self.transition_cdf = np.cumsum(self.transition_matrix, axis=1)
        self.transition_cdf[:, -1] = 1.0

    @classmethod
    def from_orbit_logits(cls, support_shape, orbit_logits):
        return cls(get_binary_spin_symmetry_data(tuple(support_shape)), orbit_logits)

    def sample_next_states(self, current_state_indices, rng):
        draws = rng.rand(*current_state_indices.shape)
        cdf_rows = self.transition_cdf[current_state_indices]
        return np.argmax(draws[..., None] <= cdf_rows, axis=-1).astype(np.int64)

    def decode_states(self, state_indices):
        return self.symmetry_data.state_bits[state_indices]

    def encode_tile_views(self, tile_views, support_offsets=None):
        if support_offsets is None:
            support_offsets = self.symmetry_data.support_offsets

        state_indices = np.zeros_like(next(iter(tile_views.values())), dtype=np.int64)
        for bit_index, offset in enumerate(support_offsets):
            state_indices |= tile_views[offset].astype(np.int64) << bit_index
        return state_indices
