import numpy as np


####################################################################################################
# Exercise 1: DFT

def dft_matrix(n: int) -> np.ndarray:
    """
    Construct DFT matrix of size n.

    Arguments:
    n: size of DFT matrix

    Return:
    F: DFT matrix of size n

    Forbidden:
    - numpy.fft.*
    """
    # TODO: initialize matrix with proper size
    F = np.zeros((n, n), dtype='complex128')

    # TODO: create principal term for DFT matrix

    # TODO: fill matrix with values
    for i in range(n):
        for j in range(n):
            F[i][j] = np.exp(-(2 * np.pi * 1j * (i * j) / n))

    # TODO: normalize dft matrix
    F *= (1 / np.sqrt(n))

    return F


def is_unitary(matrix: np.ndarray) -> bool:
    """
    Check if the passed in matrix of size (n times n) is unitary.

    Arguments:
    matrix: the matrix which is checked

    Return:
    unitary: True if the matrix is unitary
    """
    unitary = True
    # TODO: check that F is unitary, if not return false
    tmp = matrix.conjugate().transpose()

    res = np.dot(matrix, tmp)
    identity = np.identity(matrix.shape[0])
    if not np.allclose(identity, res):
        unitary = False

    return unitary


def create_harmonics(n: int = 128) -> (list, list):
    """
    Create delta impulse signals and perform the fourier transform on each signal.

    Arguments:
    n: the length of each signal

    Return:
    sigs:  list of np.ndarrays that store the delta impulse signals
    fsigs: list of np.ndarrays with the fourier transforms of the signals
    """

    # list to store input signals to DFT
    sigs = []
    # Fourier-transformed signals
    fsigs = []

    # TODO: create signals and extract harmonics out of DFT matrix
    matrix_dft = dft_matrix(n)
    sigs = np.identity(n)
    fsigs = np.dot(matrix_dft, sigs)

    return list(sigs), list(fsigs)


####################################################################################################
# Exercise 2: FFT

def shuffle_bit_reversed_order(data: np.ndarray) -> np.ndarray:
    """
    Shuffle elements of data using bit reversal of list index.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    data: shuffled data array
    """

    # TODO: implement shuffling by reversing index bits
    n = len(data)
    shuffled_data = np.zeros(n, dtype='complex128')
    for i in range(n):
        bin_i = bin(i)
        pure_bin_i = bin_i[2:]
        pure_bin_i = ''.join(('0' * (len(bin(n - 1)[2:]) - len(pure_bin_i)), pure_bin_i))
        rev_pure_bin_i = pure_bin_i[::-1]
        j = int(str(rev_pure_bin_i), 2)
        shuffled_data[j] = data[i]
    data = shuffled_data

    return data


def fft(data: np.ndarray) -> np.ndarray:
    """
    Perform real-valued discrete Fourier transform of data using fast Fourier transform.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    fdata: Fourier transformed data

    Note:
    This is not an optimized implementation but one to demonstrate the essential ideas
    of the fast Fourier transform.

    Forbidden:
    - numpy.fft.*
    """

    fdata = np.asarray(data, dtype='complex128')
    n = fdata.size

    # check if input length is power of two
    if not n > 0 or (n & (n - 1)) != 0:
        raise ValueError

    # TODO: first step of FFT: shuffle data
    shuffled_data = shuffle_bit_reversed_order(data)

    # TODO: second step, recursively merge transforms
    for m in range(int(np.log2(n))):
        for k in range(2**m):
            omega = np.exp(-2j * np.pi * k / 2**(m+1))
            for i in range(k, n, 2**(m+1)):
                j = i + 2**m
                p = omega * shuffled_data[j]
                shuffled_data[j] = shuffled_data[i] - p
                shuffled_data[i] += p

    # TODO: normalize fft signal
    shuffled_data /= np.sqrt(n)

    return shuffled_data


def generate_tone(f: float = 261.626, num_samples: int = 44100) -> np.ndarray:
    """
    Generate tone of length 1s with frequency f (default mid C: f = 261.626 Hz) and return the signal.

    Arguments:
    f: frequency of the tone

    Return:
    data: the generated signal
    """

    # sampling range
    x_min = 0.0
    x_max = 1.0

    data = np.zeros(num_samples)

    # TODO: Generate sine wave with proper frequency
    data_points = np.linspace(x_min, x_max, num_samples)
    data = np.sin(2 * np.pi * f * data_points)

    return data


def low_pass_filter(adata: np.ndarray, bandlimit: int = 1000, sampling_rate: int = 44100) -> np.ndarray:
    """
    Filter high frequencies above bandlimit.

    Arguments:
    adata: data to be filtered
    bandlimit: bandlimit in Hz above which to cut off frequencies
    sampling_rate: sampling rate in samples/second

    Return:
    adata_filtered: filtered data
    """

    # translate bandlimit from Hz to dataindex according to sampling rate and data size
    bandlimit_index = int(bandlimit * adata.size / sampling_rate)

    # TODO: compute Fourier transform of input data
    transformed_data = np.fft.fft(adata)

    # TODO: set high frequencies above bandlimit to zero, make sure the almost symmetry of the transform is respected.
    for i in range(len(transformed_data)):
        if bandlimit_index < i < (transformed_data.size - bandlimit_index):
            transformed_data[i] = 0

    # TODO: compute inverse transform and extract real component
    adata_filtered = np.zeros(adata.shape[0])
    adata_filtered = np.fft.ifft(transformed_data)

    return adata_filtered


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
