from abc import ABC

import numpy as np

from tools.filters.filter import Filter
from tools import stat_characteristics as sc, ploter


class AlphaBeta(Filter, ABC):

    def __init__(self):
        super().__init__()

    def filter(self, sample, k_max):
        self._sample = sample
        print(len(self._sample))
        t = 1
        v_previous = (sample[1] - sample[0]) / t
        x_predicted = sample[0] + v_previous
        alpha = 2 * (2 * 1 - 1) / 1 * (1 + 1)
        beta = 6 / 1 * (1 + 1)
        self._filtered_sample.append((sample[0] + alpha * (sample[0] - x_predicted)))
        for k in range(1, len(self._sample)):
            print(k)
            x = x_predicted + alpha * (sample[k] - x_predicted)
            self._filtered_sample.append(x)
            x_previous = x
            v_predicted = v_previous
            v = v_predicted + (beta / t) * (sample[k] - x_predicted)
            self._filtered_speed_list.append(v)
            v_previous = v
            x_predicted = x_previous + t * v_predicted
            if k >= k_max:
                alpha = (2 * (2 * k_max - 1)) / (k_max * (k_max + 1))
                beta = 6 / (k_max * (k_max + 1))
            else:
                alpha = (2 * (2 * k - 1)) / (k * (k + 1))
                beta = 6 / (k * (k + 1))
        return self._filtered_sample

    def info(self):
        print('\033[94mAB filtered sample stat characteristics:\033[0m')
        sc.stat_characteristics(self._filtered_sample, print_flag=True)
        ploter.two_plots(self._sample, 'sample', self._filtered_sample, 'alpha-beta-filter')
