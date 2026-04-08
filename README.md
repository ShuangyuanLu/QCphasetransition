# QC Phase Transition

Monte Carlo code for binary-spin models on a square lattice, with a newer
JSON-driven backend intended to support external AI search over update rules and
parameters.

## Current Scope

The newer backend currently targets:

- binary spins `0/1`
- square lattice with periodic boundary conditions
- translationally symmetric local updates
- support shapes `[1, 1]`, `[1, 2]`, `[2, 1]`, `[2, 2]`
- symmetry-constrained local kernels parameterized by `theta`

The repo still contains older handwritten Monte Carlo code for Ising and clock
experiments. That code is kept as a baseline and for reference. The new engine
is the path intended for future automated search.

## Main Files

- `periodic_mc.py`: core simulation engine and JSON-safe spec interface
- `binary_spin_symmetry.py`: symmetry reduction and compiled local transition kernels
- `mc_average.py`: correlated Monte Carlo mean/error estimator
- `mc_analysis.py`: burn-in, uncertainty, and stability analysis
- `evaluate_spec_cli.py`: JSON CLI entrypoint for external projects
- `main_periodic_general.py`: example of the new general backend
- `main_periodic_ising.py`: example of the newer framework on the legacy Ising-style schedule
- `main_compare_ising.py`: exact regression check between legacy Ising and the new engine
- `ENGINE_INTERFACE.md`: frozen spec/result contract for the external AI project

## Requirements

The current code assumes Python 3 and NumPy.

Example environment used during development:

```bash
/home/shuan/cupy-env/bin/python
```

The new backend does not require CuPy.

## Quick Start

Run the general symmetric-kernel example:

```bash
/home/shuan/cupy-env/bin/python main_periodic_general.py
```

Run the Ising regression check:

```bash
/home/shuan/cupy-env/bin/python main_compare_ising.py
```

## Backend Interface

The new engine is designed to be called from another project.

The intended flow is:

1. Build a JSON spec.
2. Call the backend.
3. Read a JSON result.

CLI usage:

```bash
/home/shuan/cupy-env/bin/python evaluate_spec_cli.py spec.json
/home/shuan/cupy-env/bin/python evaluate_spec_cli.py spec.json -o result.json
```

The full input/output contract is documented in:

- `ENGINE_INTERFACE.md`

## Minimal Example Spec

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

## Result Structure

The backend returns a JSON-safe result containing:

- `n_compiled_updates`
- `compiled_updates`
- `n_measurements`
- `measurement_stride_periods`
- `summary`
- `analysis`

Optional fields:

- `measurements`
- `final_state`

The `analysis` block contains per-observable uncertainty and burn-in estimates,
including:

- `stable`
- `burn_in_index`
- `mean`
- `stderr`
- `tau_int`
- `n_eff`

## Notes

- A single period block with several `shifts` expands into several compiled updates.
- `theta` length depends on the support shape:
  - `[1, 1]` -> `2`
  - `[1, 2]` -> `6`
  - `[2, 1]` -> `6`
  - `[2, 2]` -> `31`
- Current observer support is still minimal and centered on `m`, `m_2`, `m_4`.

## Status

What is already in place:

- JSON-safe spec/result interface
- CLI backend for external search code
- symmetry-compiled update kernels
- lightweight uncertainty/burn-in analysis
- Ising regression check against the legacy implementation

What is expected to live outside this repo:

- AI search / optimization logic
- candidate generation and scoring policy
- experiment orchestration across many specs
