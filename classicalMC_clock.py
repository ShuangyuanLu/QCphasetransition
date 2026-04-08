from classicalMC import classicalMC
import numpy as np
import matplotlib.pyplot as plt
import shelve


class classicalMC_clock(classicalMC):
    def __init__(self, parameters, hamiltonian_parameters):
        classicalMC.__init__(self, parameters, hamiltonian_parameters)
        self.n_clock = 5
        self.p_align = hamiltonian_parameters['p_align']
        self.flux = None

    def update_onsite(self):
        random_array_0 = np.random.rand(self.N_sample, self.L, self.L)
        random_array = random_array_0 < self.p_flip
        spin_change = random_array_0 / self.p_flip * (self.n_clock - 1)
        spin_change = np.trunc(spin_change).astype(np.int8) + 1

        self.states[:, 0: self.L, 0: self.L] = self.states[:, 0: self.L, 0: self.L] + np.where(random_array, spin_change, 0)
        self.fresh(1, 1)
        self.states = self.states % self.n_clock

    def update_interaction(self, row, column):
        if (row + column) % 2 == 0:     # horizontal interaction
            row_shift = 0
            column_shift = 1
        else:       # vertical interaction
            row_shift = 1
            column_shift = 0
        difference = (self.states[:, row: row + self.L: row_shift + 1, column: column + self.L: column_shift + 1] - self.states[:, row + row_shift: row + row_shift + self.L: row_shift + 1, column + column_shift: column + column_shift + self.L: column_shift + 1])
        difference = difference % self.n_clock
        random_array_0 = np.random.rand(self.N_sample, self.L // (row_shift + 1), self.L // (column_shift + 1))

        spin_change_1 = np.zeros_like(random_array_0, dtype=np.int8)
        spin_change_2 = np.zeros_like(random_array_0, dtype=np.int8)
        spin_change_1[(difference == 2) & (random_array_0 < self.p_align)] = -1
        spin_change_2[(difference == 2) & (random_array_0 < self.p_align)] = 1
        spin_change_1[(difference == 3) & (random_array_0 < self.p_align)] = 1
        spin_change_2[(difference == 3) & (random_array_0 < self.p_align)] = -1

        spin_change_1[(difference == 1) & (random_array_0 < self.p_align / 2)] = -1
        spin_change_1[(difference == 4) & (random_array_0 < self.p_align / 2)] = 1
        spin_change_2[(difference == 1) & (self.p_align / 2 < random_array_0) & (random_array_0 < self.p_align)] = 1
        spin_change_2[(difference == 4) & (self.p_align / 2 < random_array_0) & (random_array_0 < self.p_align)] = -1

        self.states[:, row: row + self.L: row_shift + 1, column: column + self.L: column_shift + 1] += spin_change_1
        self.states[:, row + row_shift: row + row_shift + self.L: row_shift + 1, column + column_shift: column + column_shift + self.L: column_shift + 1] += spin_change_2

        self.fresh(1 - row, 1 - column)
        self.states = self.states % self.n_clock

    def measure(self):
        exp_values = np.exp(1j * np.arange(self.n_clock) * 2 * np.pi / self.n_clock)
        cos_values = np.cos(np.arange(self.n_clock) * 2 * np.pi / self.n_clock)
        #states_complex = exp_values[self.states[:, 0: self.L, 0: self.L]]
        #m_list = np.abs(np.sum(states_complex, axis=(1, 2))) / (self.L * self.L)
        #self.measurement["m"].append(np.sum(m_list) / self.N_sample)
        #self.measurement["m_2"].append(np.sum(np.square(m_list)) / self.N_sample)
        #self.measurement["sus"].append(np.sum(np.square(m_list)) / self.N_sample - (np.sum(m_list) / self.N_sample) ** 2)
        g_L_2 = np.sum(cos_values[np.mod(self.states[:, self.L // 2: self.L, 0: self.L] - self.states[:, 0: self.L // 2, 0: self.L], self.n_clock)]) * 2
        g_L_4 = np.sum(cos_values[np.mod(self.states[:, self.L // 4: self.L, 0: self.L] - self.states[:, 0: self.L  * 3 // 4, 0: self.L], self.n_clock)])
        g_L_4 += np.sum(cos_values[np.mod(self.states[:, 0: self.L // 4, 0: self.L] - self.states[:, self.L  * 3 // 4: self.L, 0: self.L], self.n_clock)])
        self.measurement["corr"].append(g_L_2 / g_L_4)

    def calculate_flux(self):
        phase_diff = np.array([0, 1, 2, -2, -1])
        phase_diff_0 = phase_diff[np.mod(self.states[:, 1: self.L + 1, 0: self.L] - self.states[:, 0: self.L, 0: self.L], self.n_clock)]
        phase_diff_1 = phase_diff[np.mod(self.states[:, 1: self.L + 1, 1: self.L + 1] - self.states[:, 1: self.L + 1, 0: self.L], self.n_clock)]
        phase_diff_2 = phase_diff[np.mod(self.states[:, 0: self.L, 1: self.L + 1] - self.states[:, 1: self.L + 1, 1: self.L + 1], self.n_clock)]
        phase_diff_3 = phase_diff[np.mod(self.states[:, 0: self.L, 0: self.L] - self.states[:, 0: self.L, 1: self.L + 1], self.n_clock)]
        self.flux = np.floor_divide(phase_diff_0 + phase_diff_1 + phase_diff_2 + phase_diff_3, self.n_clock)

    def plot_flux(self):
        for i in range(10):
            plt.figure(figsize=(5, 5))
            plt.imshow(self.flux[i], cmap='viridis', interpolation='nearest')
            plt.show()

























