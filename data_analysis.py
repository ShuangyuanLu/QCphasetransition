import shelve
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def compute_binder_cumulant(folder_name):
    i_parallel = 0
    while Path(folder_name + "/data_" + str(i_parallel) + ".bak").exists():
        with shelve.open(folder_name + "/data_" + str(i_parallel)) as data_file:
            if i_parallel == 0:
                m_all = data_file['m']
                m_2_all = data_file['m_2']
                m_4_all = data_file['m_4']
            else:
                m_all += data_file['m']
                m_2_all += data_file['m_2']
                m_4_all += data_file['m_4']
        i_parallel += 1

    m_all = m_all / i_parallel
    m_2_all = m_2_all / i_parallel
    m_4_all = m_4_all / i_parallel

    plt.plot(m_all)
    plt.plot(m_2_all)
    plt.plot(m_4_all)
    binder_cumulant = 1 - m_4_all / (3 * np.square(m_2_all))
    plt.plot(binder_cumulant)
    print(np.mean(binder_cumulant[len(binder_cumulant) // 2:]))
    plt.show()
