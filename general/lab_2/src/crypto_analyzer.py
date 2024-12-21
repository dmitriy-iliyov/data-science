import numpy as np
import math

from lab_2.src.tools import ploter, distributions, lsm_package as lp
from lab_2.src.tools.cleaners import lsm_cleaner, medium_cleaner, sliding_window_cleaner
from lab_2.src.tools.filters import alpha_beta, alpha_beta_gamma


class CryptoAnalyzer:

    def __init__(self, sample, currency):
        self._currency = currency
        self._sample_len = len(sample)

        self._input_sample = sample
        self._cleaned_sample = None
        self._filtered_sample = None
        self._approximated_sample = None
        self._extrapolated_sample = None

        self._d = None
        self._c = None
        self._a_leftovers = None

        self._cleaners_map = {'sliding-window': sliding_window_cleaner.SlidingWindowCleaner(),
                              'medium': medium_cleaner.MediumCleaner(),
                              'lsm': lsm_cleaner.LsmCleaner()}
        self._filters_map = {'alpha-beta': alpha_beta.AlphaBeta(),
                             'alpha-beta-gamma': alpha_beta_gamma.AlphaBetaGamma()}

        print('\033[94mSample stat characteristics:\033[0m')
        self._E, self._V, self._sd = self.stat_characteristics()
        ploter.clean()
        ploter.one_plot(sample, currency)

    def clean(self, method, percent, q=None):
        cleaner = self._cleaners_map[method]
        self._cleaned_sample = cleaner.clean(self._input_sample, percent=percent)
        cleaner.info()
        return self._cleaned_sample

    def filter(self, method, k_max=np.inf):
        _filter = self._filters_map[method]
        self._filtered_sample = _filter.filter(self._cleaned_sample, k_max)
        _filter.info()
        self._cleaned_sample = self._filtered_sample
        return self._filtered_sample

    def approximate(self, d):
        self._d = self._optimize_polynomial(d)

        self._approximated_sample, self._c, self._a_leftovers = lp.lsm_approximation(self._cleaned_sample, self._d,
                                                                                     leftovers_flag=True)
        print('\033[94mApproximation sample stat characteristics:\033[0m')
        self.stat_characteristics(self._approximated_sample)
        ploter.three_plots([self._input_sample, self._cleaned_sample, self._approximated_sample],
                           ['sample', 'cleaned-sample', 'trend'])
        r2 = self._r2_score()
        print('\033[94mr2 =\033[0m', r2, '; d =', self._d, '\n')
        return r2

    def _optimize_polynomial(self, d):
        best_r2 = -np.inf
        best_degree = 0
        for i in range(0, d + 1):
            self._approximated_sample, self._c, self._a_leftovers = lp.lsm_approximation(self._cleaned_sample, i,
                                                                                         leftovers_flag=True)
            r2 = self._r2_score()
            if r2 > best_r2:
                best_r2 = r2
                best_degree = i
            else:
                return best_degree
        return best_degree

    def extrapolate(self, extrapolation_len=None):
        if extrapolation_len is None:
            extrapolation_len = int(self._sample_len/2)
        self._extrapolated_sample = lp.lsm_extrapolation(self._approximated_sample, self._c, extrapolation_len)
        print('\033[94mExtrapolation sample stat characteristics:\033[0m')
        self.stat_characteristics(self._extrapolated_sample)
        ploter.two_plots(self._input_sample, 'sample', self._extrapolated_sample, 'extrapolation-data')
        return self._input_sample, self._extrapolated_sample

    def model(self, synthetic_sample_len, noise=False, anomalies=False, save=False):
        _synthetic_a_sample = np.zeros(synthetic_sample_len)
        for j in range(synthetic_sample_len):
            _synthetic_a_sample[j] = sum(c * (j ** i) for i, c in enumerate(self._c))
        if noise:
            _synthetic_noise_sample = self._add_noise(_synthetic_a_sample, synthetic_sample_len)
            ploter.two_plots(_synthetic_noise_sample, 'synthetic-n-sample',
                             lp.lsm(_synthetic_noise_sample, self._d)[0], 'synthetic-trend')
            print('\033[94mSynthetic noise sample stat characteristics:\033[0m')
            self.stat_characteristics(_synthetic_noise_sample)
            if anomalies:
                sdk = 3
                anomalies_present = 10
                anomalies_count = int((synthetic_sample_len * anomalies_present) / 100)
                _synthetic_sample = self._add_anomalies(_synthetic_noise_sample,
                                                        anomalies_count, sdk, synthetic_sample_len)
                ploter.two_plots(_synthetic_sample, 'synthetic-sample-anomalies',
                                 lp.lsm(_synthetic_sample, self._d)[0], 'synthetic-trend')
                print('\033[94mSynthetic noise + anomalies sample stat characteristics:\033[0m')
                self.stat_characteristics(_synthetic_sample)
                if save:
                    self.reset(_synthetic_sample)
                return _synthetic_sample
            return _synthetic_noise_sample
        else:
            return _synthetic_a_sample

    def _add_noise(self, _synthetic_a_sample, sample_len):
        _synthetic_noise_sample = np.zeros(sample_len)
        noise = distributions.normal_distribution(self._E, self._sd * 0.3, sample_len)
        _synthetic_noise_sample = _synthetic_a_sample + noise
        return _synthetic_noise_sample

    def _add_anomalies(self, _synthetic_noise_sample, anomalies_count, sdk, sample_len):
        _synthetic_sample = _synthetic_noise_sample.copy()
        SSAV = np.random.normal(self._E, (sdk * self._sd), anomalies_count)
        uniform = distributions.uniform_distribution(min(self._input_sample), max(self._input_sample), sample_len,
                                                     anomalies_count)
        for i in range(anomalies_count):
            k = int(uniform[i])
            _synthetic_sample[k] = _synthetic_sample[k] + SSAV[i]
        return _synthetic_sample

    def stat_characteristics(self, sample=None):
        if sample is None:
            sample = self._input_sample
        E = np.median(sample)
        V = np.var(sample)
        sd = math.sqrt(V)
        print('sample len =', len(sample))
        print('math expectation =', E)
        print('variance =', V)
        print('standard deviation =', sd, '\n')
        return E, V, sd

    def _r2_score(self):
        numerator = 0
        denominator = 0
        for i in range(self._sample_len):
            numerator += self._a_leftovers[i]**2
            denominator += (self._input_sample[i] - self._E) ** 2
        r2 = 1 - numerator/denominator
        return r2

    def reset(self, sample, currency=None):
        self._input_sample = sample
        self._sample_len = len(sample)
        if currency:
            self._currency = currency
        self._cleaners_map = {'sliding-window': sliding_window_cleaner.SlidingWindowCleaner(),
                              'medium': medium_cleaner.MediumCleaner(),
                              'lsm': lsm_cleaner.LsmCleaner()}
        self._filters_map = {'alpha-beta': alpha_beta.AlphaBeta(),
                             'alpha-beta-gamma': alpha_beta_gamma.AlphaBetaGamma()}

    def print_cleand_sample(self):
        ploter.one_plot(self._cleaned_sample, 'cleaned-sample')
