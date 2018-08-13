import numpy as np
from typing import Optional, Tuple
import scipy.io as matloader


def generate_sample(filename, batch_size: int = 4, predict: int = 50, samples: int = 100,
                    test_set: list = [0], start_from: int = -1, test: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates data samples.

    :param filename: mat file name
    :param batch_size: The number of time series to generate.
    :param predict: The number of future samples to generate.
    :param samples: The number of past (and current) samples to generate.
    :param start_from: Offset
    :return: Tuple that contains the past times and values as well as the future times and values. In all outputs,
             each row represents one time series of the batch.
    """

    mat = matloader.loadmat(filename)

    T = np.empty((batch_size, samples))
    Y = np.empty((batch_size, samples))
    FT = np.empty((batch_size, predict))
    FY = np.empty((batch_size, predict))

    arr_length = []

    training_list = set([0, 1, 2, 3, 4]) - set(test_set)
    if test > 0:
        action_set = test_set
    else:
        action_set = training_list
    jdx = 0
    for i in action_set:
        total_data = len(mat['RoI_All'][0, i][0])
        arr_length.append(total_data)

        idx = np.random.random_integers(total_data - (samples + predict))

        if -1 < start_from < total_data - (samples + predict):
            idx = start_from

        T [jdx, :] = range(idx, idx + samples)
        Y [jdx, :] = np.transpose(mat['RoI_All'][0, i][0][idx:idx + samples])
        FT[jdx, :] = range(idx + samples, idx + samples + predict)
        FY[jdx, :] = np.transpose(mat['RoI_All'][0, i][0][idx + samples:idx + samples + predict])
        jdx += 1

    return T, Y, FT, FY, arr_length


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # noinspection PyUnresolvedReferences
    import seaborn as sns

    samples = 100
    predict = 50

    t, y, t_next, y_next, lengths = generate_sample(filename="RoIFor5Devs.mat", batch_size=1, test=True)

    n_tests = t.shape[0]
    for i in range(0, n_tests):
        plt.subplot(n_tests, 1, i+1)
        plt.plot(t[i, :], y[i, :])
        plt.plot(np.append(t[i, -1], t_next[i, :]), np.append(y[i, -1], y_next[i, :]), color='red', linestyle=':')

    plt.xlabel('time [t]')
    plt.ylabel('signal')
    plt.show()
