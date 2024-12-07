import numpy as np
import math

from lab_1 import ploter
from lab_1 import distributions as d


class CryptoAnalyzer:
    def __init__(self, sample, currency):
        self.sample = sample
        self.currency = currency
        print('Sample stat characteristics:')
        self.E, self.V, self.sd = self.stat_characteristics()

        ploter.one_plot(sample, currency)

        self._a_sample = None
        self._c = None
        self._e_sample = None

        self._synthetic_a_sample = None
        self._synthetic_noise_sample = None
        self._synthetic_sample = None

    def model(self, n, noise=False, anomalies=False, anomalies_present=10):
        self._synthetic_a_sample = np.zeros(n)
        for j in range(n):
            self._synthetic_a_sample[j] = sum(c * (j ** i) for i, c in enumerate(self._c))
        if noise:
            ploter.two_plots(self._add_noise(n), 'synthetic-n-sample',
                             self._LSM(self._synthetic_noise_sample)[0], 'synthetic-trend')
            print('Synthetic noise sample stat characteristics:')
            self.stat_characteristics(self._synthetic_noise_sample)
            if anomalies:
                sdk = 3
                anomalies_count = int((n * anomalies_present) / 100)
                ploter.two_plots(self._add_anomalies(anomalies_count, sdk, n), 'synthetic-sample-anomalies',
                                 self._LSM(self._synthetic_sample)[0], 'synthetic-trend')
                print('Synthetic noise + anomalies sample stat characteristics:')
                self.stat_characteristics(self._synthetic_sample)
                return self._synthetic_sample
            return self._synthetic_noise_sample
        else:
            return self._synthetic_a_sample

    def _add_noise(self, n):
        self._synthetic_noise_sample = np.zeros(n)
        noise = d.normal_distribution(self.E, self.sd, n)
        self._synthetic_noise_sample = self._synthetic_a_sample + noise
        return self._synthetic_noise_sample

    def _add_anomalies(self, anomalies_count, sdk, n):
        self._synthetic_sample = self._synthetic_noise_sample.copy()
        SSAV = np.random.normal(self.E, (sdk * self.sd), anomalies_count)
        uniform = d.uniform_distribution(min(self.sample), max(self.sample), n, anomalies_count)
        for i in range(anomalies_count):
            k = int(uniform[i])
            self._synthetic_sample[k] = self._synthetic_sample[k] + SSAV[i]
        return self._synthetic_sample

    def stat_characteristics(self, sample=None):
        if sample is None:
            sample = self.sample
        E = np.median(sample)
        V = np.var(sample)
        sd = math.sqrt(V)
        print('sample len = ', len(sample))
        print('math expectation = ', E)
        print('variance = ', V)
        print('standard deviation = ', sd, '\n')
        return E, V, sd

    def lsm_approximation(self, leftovers_flag=False):
        self._a_sample, self._c = self._LSM(self.sample)
        sample_len = len(self._a_sample)
        leftovers = np.zeros(sample_len)
        for i in range(sample_len):
            leftovers[i] = self.sample[i] - self._a_sample[i, 0]
        ploter.two_plots(self.sample, 'sample', self._a_sample, 'trend')
        print('Approximation sample stat characteristics:')
        self.stat_characteristics(self._a_sample)
        if leftovers_flag:
            return self._a_sample, leftovers
        return self._a_sample

    def lsm_extrapolation(self, extrapolation_sample_length):
        self._e_sample = np.zeros((len(self.sample) + extrapolation_sample_length, 1))
        if self._c is None:
            self._c = self._LSM(self.sample)[1]
        for i in range(1, len(self._e_sample)):
            self._e_sample[i, 0] = (self._c[0, 0] + self._c[1, 0] * i + (self._c[2, 0] * i * i)
                                    + (self._c[3, 0] * i ** 3) + (self._c[4, 0] * i ** 4))
        print('Extrapolation sample stat characteristics:')
        self.stat_characteristics(self._e_sample)
        ploter.one_plot(self._e_sample, 'extrapolation-data')
        return self._e_sample

    def _LSM(self, sample):
        sample_len = len(sample)
        Yin = np.zeros((sample_len, 1))
        F = np.ones((sample_len, 5))
        for i in range(sample_len):
            Yin[i, 0] = float(sample[i])
            F[i, 1] = float(i)
            F[i, 2] = float(i * i)
            F[i, 3] = float(i ** 3)
            F[i, 4] = float(i ** 4)
        FT = F.T
        FFT = FT.dot(F)
        FFTI = np.linalg.inv(FFT)
        FFTIFT = FFTI.dot(FT)
        C = FFTIFT.dot(Yin)
        approximated_sample = F.dot(C)
        return approximated_sample, C
