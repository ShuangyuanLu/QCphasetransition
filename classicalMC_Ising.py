from classicalMC import classicalMC
import numpy as np
import matplotlib.pyplot as plt



class classicalMC_Ising(classicalMC):
    def __init__(self, parameters, hamiltonian_parameters):
        classicalMC.__init__(self, parameters, hamiltonian_parameters)

    def update_onsite(self):
        random_array = np.random.rand(self.N_sample, self.L, self.L) < self.p_flip
        self.states[:, 0: self.L, 0: self.L] = self.states[:, 0: self.L, 0: self.L] ^ random_array
        self.fresh(1, 1)

    def update_interaction(self, row, column):
        ones_2 = np.ones((2, 2), dtype=np.int8)
        random_array_0 = np.random.rand(self.N_sample, self.L // 2, self.L // 2)

        random_array = np.kron(random_array_0 < 0.5, ones_2)

        cluster = (self.states[:, row: row + self.L: 2, column: column + self.L: 2] + self.states[:, row + 1: row + self.L: 2, column: column + self.L: 2]
                        + self.states[:, row: row + self.L: 2, column + 1: column + self.L: 2] + self.states[:, row + 1: row + self.L: 2, column + 1: column + self.L: 2])
        new_states = np.logical_or(np.kron(cluster == 3, ones_2), self.states[:, row: row + self.L, column: column + self.L])
        new_states = np.logical_and(np.logical_not(np.kron(cluster == 1, ones_2)), new_states)
        new_states = np.logical_xor(np.kron(cluster == 2, ones_2), new_states)
        self.states[:, row: row + self.L, column: column + self.L] = np.where(random_array, new_states, self.states[:, row: row + self.L, column: column + self.L])
        self.fresh(1 - row, 1 - column)

    def plot_spin(self):
        for i in range(50):
            #plt.imsave("data/spin_" + str(i) + ".jpg", self.states[i, 0: self.L, 0: self.L], cmap='gray_r', dpi=300)
            plt.figure(figsize=(6, 6))  # 6×6 inches
            plt.imshow(self.states[i, 0: self.L, 0: self.L], cmap='gray_r', interpolation='nearest')
            plt.axis('off')
            plt.savefig("data/spin_" + str(i) + ".png", dpi=300, bbox_inches='tight', pad_inches=0)
            #plt.show()
