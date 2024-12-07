import numpy as np
import math

from lab_1 import ploter


def uniform_distribution(a, b, n, nAV):
    random_sample = np.array([np.random.uniform(a, b) for _ in range(n)])
    ploter.hist(random_sample)
    SAV = np.zeros(nAV)
    for i in range(nAV):
        SAV[i] = math.ceil(np.random.randint(1, n))
    return SAV


def normal_distribution(E, sd, n):
    random_sample = np.array(np.random.normal(E, sd, n))
    ploter.hist(random_sample)
    return random_sample
