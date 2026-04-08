import numpy as np

from periodic_mc import (
    build_runner_from_spec,
    n_transition_orbits_for_shape,
)


if __name__ == "__main__":
    parameters = {
        "L": 12,
        "n_periods": 200,
        "measure_every_periods": 10,
        "N_sample": 1000,
        "random_seed": 0,
    }
    rule_parameters = {"p_flip": 0.05, "p_align": 0.4}
    pair_orbit_logits = np.zeros(n_transition_orbits_for_shape((1, 2)))
    rule_parameters["pair_orbit_logits"] = pair_orbit_logits

    # Swap the schedule name to try different combinations of translated local updates.
    schedule_name = "onsite_and_block"
    period_builders = {
        "onsite_and_block": lambda params: [
            {
                "name": "onsite",
                "family": "single_site_flip",
                "params": {"p_flip": params["p_flip"]},
            },
            {
                "name": "block",
                "family": "plaquette",
                "shifts": [(0, 0), (1, 0), (1, 1), (0, 1)],
            },
        ],
        "onsite_and_pair": lambda params: [
            {
                "name": "onsite",
                "family": "single_site_flip",
                "params": {"p_flip": params["p_flip"]},
            },
            {
                "name": "pair_h",
                "family": "pair_alignment",
                "support_shape": (1, 2),
                "shifts": [(0, 0), (0, 1)],
                "params": {"p_align": params["p_align"]},
            },
            {
                "name": "pair_v",
                "family": "pair_alignment",
                "support_shape": (2, 1),
                "shifts": [(0, 0), (1, 0)],
                "params": {"p_align": params["p_align"]},
            },
        ],
        "onsite_and_symmetric_pair": lambda params: [
            {
                "name": "onsite",
                "family": "single_site_flip",
                "params": {"p_flip": params["p_flip"]},
            },
            {
                "name": "sym_pair_h",
                "family": "symmetric_kernel",
                "support_shape": (1, 2),
                "shifts": [(0, 0), (0, 1)],
                "params": {"orbit_logits": params["pair_orbit_logits"]},
            },
            {
                "name": "sym_pair_v",
                "family": "symmetric_kernel",
                "support_shape": (2, 1),
                "shifts": [(0, 0), (1, 0)],
                "params": {"orbit_logits": params["pair_orbit_logits"]},
            },
        ],
    }

    runner_spec = {
        "simulation": parameters,
        "observer": {"kind": "ising_magnetization"},
        "period": period_builders[schedule_name](rule_parameters),
    }
    runner = build_runner_from_spec(runner_spec)
    measurements = runner.run()

    print(f"Schedule: {schedule_name}")
    print("Period definition:")
    for update in runner.updates:
        print(
            f"{update.name}: unit_cell={update.unit_cell_shape}, "
            f"support={update.support_offsets}, shift={update.shift}"
        )

    print()
    print(f"Recorded {len(measurements['m'])} measurements.")
    for key in ("m", "m_2", "m_4"):
        values = np.array(measurements[key])
        print(f"{key} tail mean = {np.mean(values[len(values) // 2:])}")
