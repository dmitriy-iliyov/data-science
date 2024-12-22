from abc import ABC

import numpy as np
from src.tools.cleaners.cleaner import Cleaner
from src.tools import stat_characteristics as sc, ploter


class SlidingWindowCleaner(Cleaner, ABC):

    def __init__(self):
        super().__init__()

    def clean(self, sample, percent, q=None):
        self._sample = sample
        input_sample_len = len(sample)

        first_wind_size = int(percent * input_sample_len / 100)
        first_step_sample = np.zeros(input_sample_len)
        E = np.median(sample[:first_wind_size])
        first_step_sample[:first_wind_size] = E

        for i in range(1, input_sample_len - first_wind_size + 1):
            window_sample = sample[i:i + first_wind_size]
            E = np.median(window_sample)
            first_step_sample[i:i + first_wind_size] = E

        second_wind_size = int(2 * percent * input_sample_len / 100)
        result_clean_sample = np.zeros(input_sample_len)
        E = np.median(first_step_sample[input_sample_len - second_wind_size:input_sample_len])
        result_clean_sample[input_sample_len - second_wind_size:input_sample_len] = E

        for i in range(input_sample_len - 1, second_wind_size - 1, -1):
            window_sample = first_step_sample[i - second_wind_size:i]
            E = np.median(window_sample)
            result_clean_sample[i - second_wind_size:i] = E

        self._cleaned_sample = result_clean_sample
        return self._cleaned_sample

    def info(self):
        print('\033[94mCleaned with sliding window stat characteristics:\033[0m')
        sc.stat_characteristics(self._cleaned_sample, print_flag=True)
        ploter.two_plots(self._sample, 'sample', self._cleaned_sample, 'cleaned-sample')
