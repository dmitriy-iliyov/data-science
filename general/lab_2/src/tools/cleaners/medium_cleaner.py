from abc import ABC

from lab_2.src.tools.cleaners.cleaner import Cleaner
from lab_2.src.tools import stat_characteristics as sc, ploter


class MediumCleaner(Cleaner, ABC):

    def __init__(self):
        super().__init__()

    def clean(self, sample, percent, q=3):
        self._sample = sample
        sample_len = len(sample)
        self._cleaned_sample = list(sample.values)
        wind_size = int(percent * sample_len / 100)
        _, _, standard_sd = sc.stat_characteristics(sample[:wind_size])
        self._cleaned_sample[:wind_size] = sample[:wind_size]
        for i in range(1, sample_len):
            window_sample = sample[i:i + wind_size]
            E, _, sd = sc.stat_characteristics(window_sample)
            if sd > q * standard_sd:
                self._cleaned_sample[i] = E
        return self._cleaned_sample

    def info(self):
        print('\033[94mCleaned with medium stat characteristics:\033[0m')
        sc.stat_characteristics(self._cleaned_sample, print_flag=True)
        ploter.two_plots(self._sample, 'sample', self._cleaned_sample, 'medium-cleaned-sample')
