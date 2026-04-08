from classicalMC import classicalMC
from classicalMC_Ising import classicalMC_Ising
from classicalMC_clock import classicalMC_clock
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool
import multiprocessing
import concurrent.futures
import copy
import shelve
import os
from data_analysis import compute_binder_cumulant


def run_in_parallel(classical_mc):
    classical_mc.update()
    classical_mc.save_results()
    return classical_mc

def run_parallel(parameters, hamiltonian_parameters, n_parallel):
    if parameters['N_sample'] % n_parallel != 0:
        raise ValueError('N_sample must be divisible by n_parallel')

    classical_mc_list = []
    for i in range(n_parallel):
        parameters_i = copy.deepcopy(parameters)
        parameters_i['N_sample'] = parameters['N_sample'] // n_parallel
        parameters_i['random_seed'] = i
        parameters_i['data_file_name'] = "data_" + str(i)
        classical_mc_list.append(classicalMC_Ising(parameters_i, hamiltonian_parameters))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        classical_mc_list = executor.map(run_in_parallel, classical_mc_list)


if __name__ == "__main__":
    parameters = {'L': 12, 'T': 200, 'T_measure': 10, 'N_sample': 1000, 'random_seed': 0, 'folder_name': "data/data_set_29", 'data_file_name': "data_0"}
    hamiltonian_parameters = {'p_flip': 0.05, 'p_align': 0.9}
    classical_mc = classicalMC_clock(parameters, hamiltonian_parameters)

    '''
    os.mkdir(parameters['folder_name'])
    with shelve.open(parameters['folder_name'] + "/parameters") as data_file:
        data_file['parameters'] = parameters
        data_file['hamiltonian_parameters'] = hamiltonian_parameters

    #classical_mc.update()
    #classical_mc.save_results()
    run_parallel(parameters, hamiltonian_parameters, 20)
    compute_binder_cumulant(parameters['folder_name'])
    '''
    np.random.seed(0)
    np.set_printoptions(precision=3, suppress=True)
    classical_mc.update()
    #classical_mc.calculate_flux()
    #classical_mc.plot_flux()


    #plt.plot(classical_mc.measurement["m"])
    #plt.plot(classical_mc.measurement["m_2"])
    #plt.plot(classical_mc.measurement["sus"])
    corr = classical_mc.measurement["corr"]
    plt.plot(classical_mc.measurement["corr"])
    print(sum(corr[len(corr) // 2:]) / (len(corr) / 2))
    plt.show()

    '''
    print(classical_mc.states[0, :, :])
    print(classical_mc.states[1, :, :])
    print(classical_mc.flux[0, :, :])
    print(classical_mc.flux[1, :, :])
    '''

#np.set_printoptions(threshold=sys.maxsize, linewidth=1000)
#print(classical_mc.states)


