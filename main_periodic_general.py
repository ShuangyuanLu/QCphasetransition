from periodic_mc import (
    evaluate_spec,
    make_model_spec,
    make_symmetric_kernel_update_spec,
    n_transition_orbits_for_shape,
)


if __name__ == "__main__":
    simulation_parameters = {
        "L": 12,
        "N_sample": 1000,
        "n_periods": 200,
        "measure_every_periods": 10,
        "random_seed": 0,
    }

    # Replace these theta lists with the values proposed by the future AI layer.
    theta_11 = [0.0] * n_transition_orbits_for_shape((1, 1))
    theta_12 = [0.0] * n_transition_orbits_for_shape((1, 2))
    theta_21 = [0.0] * n_transition_orbits_for_shape((2, 1))
    theta_22 = [0.0] * n_transition_orbits_for_shape((2, 2))

    analysis_parameters = {
        "min_tail": 8,
        "stability_z": 2.0,
        "target_observable": "m_2",
        "target_stderr": 0.01,
    }

    general_spec = make_model_spec(
        simulation_parameters=simulation_parameters,
        period_spec=[
            make_symmetric_kernel_update_spec("tile_11", [1, 1], [[0, 0]], theta=theta_11),
            make_symmetric_kernel_update_spec("tile_12", [1, 2], [[0, 0], [0, 1]], theta=theta_12),
            make_symmetric_kernel_update_spec("tile_21", [2, 1], [[0, 0], [1, 0]], theta=theta_21),
            make_symmetric_kernel_update_spec("tile_22", [2, 2], [[0, 0], [1, 0], [1, 1], [0, 1]], theta=theta_22),
        ],
        analysis_spec=analysis_parameters,
    )

    result = evaluate_spec(general_spec)

    print("General symmetric-kernel spec:")
    for update_spec in general_spec["period"]:
        support_shape = update_spec["support_shape"]
        n_orbits = len(update_spec["params"]["theta"])
        print(
            f"{update_spec['name']}: family={update_spec['family']}, "
            f"support_shape={support_shape}, shifts={update_spec['shifts']}, orbit_params={n_orbits}"
        )

    print()
    print(f"Compiled {result['n_compiled_updates']} translated updates in one period.")
    print(f"Recorded {result['n_measurements']} measurements.")
    for key in ("m", "m_2", "m_4"):
        print(f"{key} tail mean = {result['summary'][key]['tail_mean']}")

    print()
    print("Observable analysis:")
    for observable_name, observable_analysis in result["analysis"]["observables"].items():
        print(f"{observable_name}:")
        print(f"  stable = {observable_analysis['stable']}")
        print(f"  burn_in_index = {observable_analysis['burn_in_index']}")
        print(f"  mean = {observable_analysis['mean']}")
        print(f"  stderr = {observable_analysis['stderr']}")
        print(f"  tau_int = {observable_analysis['tau_int']}")
        if "recommended_additional_periods" in observable_analysis:
            print(f"  recommended_additional_periods = {observable_analysis['recommended_additional_periods']}")
