import math

import numpy as np


def stat_characteristics(sample, print_flag=False):
    E = np.median(sample)
    V = np.var(sample)
    sd = math.sqrt(V)
    if print_flag:
        print('sample len =', len(sample))
        print('math expectation =', E)
        print('variance =', V)
        print('standard deviation =', sd, '\n')
    return E, V, sd
