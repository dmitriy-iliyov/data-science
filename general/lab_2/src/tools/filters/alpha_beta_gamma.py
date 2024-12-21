from abc import ABC

from lab_2.src.tools.filters.filter import Filter
from lab_2.src.tools import stat_characteristics as sc, ploter


class AlphaBetaGamma(Filter, ABC):

    def __init__(self):
        super().__init__()

    def filter(self, sample, k_max):
        pass

    def info(self):
        print('\033[94mABG filtered sample stat characteristics:\033[0m')
        sc.stat_characteristics(self._filtered_sample)
        ploter.two_plots(self._sample, 'sample', self._filtered_sample, 'abg-filtered-sample')