import numpy as np


def lsm_approximation(sample, d, leftovers_flag=False):
    _a_sample, _c = lsm(sample, d)
    sample_len = len(_a_sample)
    _a_leftovers = np.zeros(sample_len)
    for i in range(sample_len):
        _a_leftovers[i] = sample[i] - _a_sample[i, 0]
    if leftovers_flag:
        return _a_sample, _c, _a_leftovers
    return _a_sample, _c


def lsm_extrapolation(sample, c, extrapolation_length):
    sample_len = len(sample)
    _e_sample = np.zeros((sample_len + extrapolation_length, 1))
    for i in range(sample_len):
        _e_sample[i, 0] = sample[i]
    for i in range(sample_len, len(_e_sample)):
        _e_sample[i, 0] = sum(c * (i ** a) for a, c in enumerate(c))
    return _e_sample


def lsm(sample, d):
    sample_len = len(sample)
    Yin = np.zeros((sample_len, 1))
    F = np.ones((sample_len, d))
    for i in range(sample_len):
        Yin[i, 0] = float(sample[i])
        power = [float(i ** a)for a in range(1, d)]
        F[i, 1:] = power
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    approximated_sample = F.dot(C)
    return approximated_sample, C
