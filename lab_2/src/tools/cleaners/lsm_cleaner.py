from abc import ABC

from tools import stat_characteristics as sc, lsm_package as lp
from tools.cleaners.cleaner import Cleaner


class LsmCleaner(Cleaner, ABC):

    def __init__(self):
        super().__init__()

    def clean(self, sample, percent, q=None):
        sample_len = len(sample)
        wind_size = int(percent * sample_len / 100)
        for i in range(0, sample_len):
            window_sample = sample[i:i + wind_size]
            approximated_sample = lp.lsm(window_sample, 9)[0]
            E, V, sd = sc.stat_characteristics(approximated_sample)
            if i + wind_size > sample_len:
                break

    def info(self):
        pass
