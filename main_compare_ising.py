import numpy as np

from classicalMC_Ising import classicalMC_Ising
from periodic_mc import build_ising_runner


def compare_series(name, old_values, new_values):
    old_array = np.array(old_values)
    new_array = np.array(new_values)
    exact_match = np.array_equal(old_array, new_array)
    max_abs_diff = np.max(np.abs(old_array - new_array)) if old_array.size else 0.0
    print(f"{name}: exact_match={exact_match}, max_abs_diff={max_abs_diff}")


if __name__ == "__main__":
    parameters = {
        "L": 12,
        "T": 200,
        "T_measure": 10,
        "N_sample": 1000,
        "random_seed": 0,
        "folder_name": "unused",
        "data_file_name": "unused",
    }
    hamiltonian_parameters = {"p_flip": 0.05}

    old_mc = classicalMC_Ising(parameters, hamiltonian_parameters)
    old_mc.update()

    new_runner = build_ising_runner(parameters, hamiltonian_parameters)
    new_measurement = new_runner.run()

    print("Measurement comparison:")
    for key in ("m", "m_2", "m_4"):
        compare_series(key, old_mc.measurement[key], new_measurement[key])

    states_match = np.array_equal(old_mc.states, new_runner.state.sites)
    print()
    print(f"Final state exact_match={states_match}")
