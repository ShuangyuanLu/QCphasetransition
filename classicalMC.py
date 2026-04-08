import numpy as np
import math
import matplotlib.pyplot as plt
import shelve
from abc import ABC, abstractmethod

class classicalMC:
    def __init__(self, parameters, hamiltonian_parameters):
        self.parameters = parameters
        self.hamiltonian_parameters = hamiltonian_parameters
        self.L = parameters['L']
        self.T = parameters['T']
        self.T_measure = parameters['T_measure']
        self.N_sample = parameters['N_sample']
        self.p_flip = hamiltonian_parameters['p_flip']      # sin(t) ** 2

        self.states = np.zeros((self.N_sample, self.L + 1, self.L + 1), dtype=np.int8)
        self.measurement = {"m": [], "m_2": [], "m_4": [], "sus": [], "corr": []}

        self.folder_name = self.parameters['folder_name']
        self.data_file_name = self.parameters['data_file_name']
        self.random_seed = self.parameters['random_seed']

    def update(self):
        np.random.seed(self.random_seed)
        for i in range(self.T):
            self.update_onsite()
            self.update_interaction(0, 0)
            self.update_interaction(1, 0)
            self.update_interaction(1, 1)
            self.update_interaction(0, 1)
            if i % self.T_measure == 0:
                self.measure()

    @abstractmethod
    def update_onsite(self):
        pass

    @abstractmethod
    def update_interaction(self, row, column):
        pass

    def fresh(self, row, column):   # row and column are 0, 1, None
        if row == 0:
            self.states[:, 0, :] = self.states[:, self.L, :]
        elif row == 1:
            self.states[:, self.L, :] = self.states[:, 0, :]

        if column == 0:
            self.states[:, :, 0] = self.states[:, :, self.L]
        elif column == 1:
            self.states[:, :, self.L] = self.states[:, :, 0]

        #self.states[:, self.L * row, :] = self.states[:, self.L * (1 - row), :]
        #self.states[:, :, self.L * column] = self.states[:, :, self.L * (1 - column)]

    def measure(self):
        m_list = 1 - 2 * np.sum(self.states[:, 0: self.L, 0:self.L], axis=(1, 2)) / (self.L * self.L)
        self.measurement["m"].append(np.sum(m_list) / self.N_sample)
        self.measurement["m_2"].append(np.sum(np.square(m_list)) / self.N_sample)
        self.measurement["m_4"].append(np.sum(np.power(m_list, 4)) / self.N_sample)

    def save_results(self):
        with shelve.open(self.folder_name + "/" + self.data_file_name) as data_file:
            data_file['m'] = np.array(self.measurement["m"])
            data_file['m_2'] = np.array(self.measurement["m_2"])
            data_file['m_4'] = np.array(self.measurement["m_4"])

