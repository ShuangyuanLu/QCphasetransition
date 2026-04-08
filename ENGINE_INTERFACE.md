# Engine Interface

This repo is the Monte Carlo backend. The future AI project should treat it as a
black-box engine with this contract:

1. Build a JSON spec.
2. Run `evaluate_spec_cli.py` or call `evaluate_spec(spec)`.
3. Read the JSON result.

The recommended AI-facing update family is `symmetric_kernel`.

## Input Spec

The spec must be JSON-safe:

- objects: `dict`
- arrays: `list`
- scalars: `str`, `int`, `float`, `bool`, `null`

Do not rely on NumPy arrays, tuples, or custom Python objects at the interface.

### Top-Level Shape

```json
{
  "simulation": { "...": "..." },
  "observer": { "...": "..." },
  "period": [ "...", "..." ],
  "analysis": { "...": "..." }
}
```

### `simulation`

Required fields:

- `L`: lattice size
- `N_sample`: number of walkers
- `random_seed`: RNG seed

You must specify one of:

- `n_periods`
- `T`

You must specify one of:

- `measure_every_periods`
- `T_measure`

Recommended form:

```json
{
  "L": 12,
  "N_sample": 1000,
  "n_periods": 200,
  "measure_every_periods": 10,
  "random_seed": 0
}
```

### `observer`

Currently supported stable value:

```json
{
  "kind": "ising_magnetization"
}
```

This observer returns the time series:

- `m`
- `m_2`
- `m_4`

Each checkpoint value is already averaged over all walkers.

### `period`

`period` is an ordered list of update blocks. Each block describes one local
update family together with its support shape, its translated shifts, and its
parameters.

Recommended stable block type for AI search:

```json
{
  "name": "tile_12",
  "family": "symmetric_kernel",
  "support_shape": [1, 2],
  "shifts": [[0, 0], [0, 1]],
  "params": {
    "theta": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  }
}
```

Notes:

- `support_shape` defines the local neighborhood geometry.
- `shifts` defines which translated tilings of that geometry are applied during
  one period.
- One block with multiple `shifts` expands into multiple compiled updates.

#### Supported `support_shape`

- `[1, 1]`
- `[1, 2]`
- `[2, 1]`
- `[2, 2]`

#### Required `theta` Length

For `family = "symmetric_kernel"`, `params["theta"]` must have length:

- `[1, 1]` -> `2`
- `[1, 2]` -> `6`
- `[2, 1]` -> `6`
- `[2, 2]` -> `31`

These are the symmetry-reduced transition-orbit parameters for the local kernel.

#### Shift Semantics

Distinct useful shifts are taken modulo the support shape:

- `[1, 1]`: `[[0, 0]]`
- `[1, 2]`: `[[0, 0], [0, 1]]`
- `[2, 1]`: `[[0, 0], [1, 0]]`
- `[2, 2]`: `[[0, 0], [1, 0], [0, 1], [1, 1]]`

A block does not have to include all of them. The AI may choose any subset or
order by changing the `period` list.

### `analysis`

Optional. This controls the lightweight post-processing on the observable time
series.

Supported fields:

- `min_tail`
- `first_window_fraction`
- `last_window_fraction`
- `stability_z`
- `target_stderr`
- `target_stderr_by_observable`
- `target_observable`

Example:

```json
{
  "min_tail": 8,
  "stability_z": 2.0,
  "target_observable": "m_2",
  "target_stderr": 0.01
}
```

`target_observable` does not limit analysis to one variable. All observables are
analyzed. It only selects the primary observable for the stopping
recommendation.

## Result Object

`evaluate_spec(spec)` returns a JSON-safe result.

Guaranteed top-level fields:

- `n_compiled_updates`
- `compiled_updates`
- `n_measurements`
- `measurement_stride_periods`
- `summary`
- `analysis`

Optional fields:

- `measurements` if raw measurement output is included
- `final_state` if requested

### `compiled_updates`

Each compiled update contains:

- `name`
- `unit_cell_shape`
- `support_offsets`
- `shift`

This is useful because one period block with several shifts expands into several
compiled updates.

### `summary`

For each observable:

- `n`
- `mean`
- `tail_mean`
- `last`

### `analysis`

Guaranteed fields:

- `series_definition`
- `measurement_stride_periods`
- `config`
- `observables`

If `target_observable` is present and valid, also:

- `primary_observable`
- `primary_analysis`

For each observable in `analysis["observables"]`, the backend returns:

- `series_length`
- `stable`
- `burn_in_index`
- `burn_in_period`
- `tail_length`
- `mean`
- `stderr`
- `tau_int`
- `n_eff`
- `stability_z_score`
- `stability_threshold`
- `stability_reason`
- `first_window_length`
- `last_window_length`

If a target stderr is configured, it also returns:

- `target_stderr`
- `recommended_total_measurements`
- `recommended_additional_measurements`
- `recommended_additional_periods`

## CLI

Evaluate a spec file and print JSON:

```bash
/home/shuan/cupy-env/bin/python evaluate_spec_cli.py spec.json
```

Write the result to a file:

```bash
/home/shuan/cupy-env/bin/python evaluate_spec_cli.py spec.json -o result.json
```

Useful flags:

- `--no-measurements`
- `--include-final-state`
- `--tail-start-fraction`
- `--indent`

## Minimal Example

```json
{
  "simulation": {
    "L": 12,
    "N_sample": 1000,
    "n_periods": 200,
    "measure_every_periods": 10,
    "random_seed": 0
  },
  "observer": {
    "kind": "ising_magnetization"
  },
  "period": [
    {
      "name": "tile_11",
      "family": "symmetric_kernel",
      "support_shape": [1, 1],
      "shifts": [[0, 0]],
      "params": {
        "theta": [0.0, 0.0]
      }
    },
    {
      "name": "tile_22",
      "family": "symmetric_kernel",
      "support_shape": [2, 2],
      "shifts": [[0, 0], [1, 0], [0, 1], [1, 1]],
      "params": {
        "theta": [
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
      }
    }
  ],
  "analysis": {
    "min_tail": 8,
    "stability_z": 2.0,
    "target_observable": "m_2",
    "target_stderr": 0.01
  }
}
```

For a runnable example inside this repo, see `main_periodic_general.py`.
