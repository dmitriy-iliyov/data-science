import numpy as np

from lab_2_fce.lab_2.tools import data_processing as dp, plotter, filer
from models import fnn, cnn, enn


def print_resulting_plots(epochs, test_len):
    t1, t1_tests_count = dp.foo(1, epochs, test_len)
    t2, t2_tests_count = dp.foo(2, epochs, test_len)
    print(t1_tests_count)
    print(t2_tests_count)
    for network in t1.keys():
        for sample in t1[network]:
            t1[network][sample] = sorted(t1[network][sample], reverse=True)
            t2[network][sample] = sorted(t2[network][sample], reverse=True)
    for network in t1.keys():
        plotter.results_plots(network,
                             t1[network]['tp1'], t1[network]['ta1'], t1[network]['tp2'], t1[network]['ta2'],
                             t2[network]['tp1'], t2[network]['ta1'], t2[network]['tp2'], t2[network]['ta2'])


class NetworkAnalyzer:

    def __init__(self):
        self._train_d = None
        self._td2 = None
        self._td3 = None
        self._train_a = None
        self._a2 = None
        self._a3 = None
        self._network_list = None
        self._network_list_2 = None

    def fit_networks(self, epochs, learning_rate, batch_size, n):
        self._network_list = [fnn.FNN(2, 1, 10),
                              cnn.CNN(2, 1, 20),
                              enn.ENN(2, 1, 15)]
        self._network_list_2 = [fnn.FNN(2, 1, 20),
                                cnn.CNN(2, 2, 10),
                                enn.ENN(2,3, 5)]

        self._train_d, self._train_a, self._td2, self._a2 = dp.prepared_data(-1, 1, n)
        test_size = int(n/5)
        self._td3, self._a3 = dp.prepared_data(-1, 1, test_size, False)

        self._fit_part(self._network_list, epochs, learning_rate, batch_size)
        self._fit_part(self._network_list_2, epochs, learning_rate, batch_size)

    def _fit_part(self, network_list, epochs, learning_rate, batch_size):
        _test_data = {}
        for network in network_list:
            if isinstance(network, enn.ENN):
                time_series = 1
                _fit_data = network.fit(self._train_d.reshape(-1, time_series, 2), self._train_a, epochs, learning_rate,
                                        batch_size)
                predictions_1 = [i for i in network.predict(self._td2.reshape(-1, time_series, 2))]
                predictions_2 = [i for i in network.predict(self._td3.reshape(-1, time_series, 2))]
            else:
                _fit_data = network.fit(self._train_d, self._train_a, epochs, learning_rate, batch_size)
                predictions_1 = [network.predict(i)[0] for i in self._td2]
                predictions_2 = [network.predict(i)[0] for i in self._td3]
            count = 0
            deviation_2 = max(self._a2) / 100
            _test_data['test_predicts_1'] = predictions_1
            _test_data['test_answers_1'] = self._a2.tolist()
            for i, p in enumerate(predictions_1):
                if abs(p - self._a2[i]) < deviation_2:
                    count += 1
            _test_data['mse_1'] = float(np.mean((self._a2 - predictions_1) ** 2))
            _test_data['score_1'] = count / len(self._a2)
            count = 0
            deviation_3 = max(self._a3) / 100
            _test_data['test_predicts_2'] = predictions_2
            _test_data['test_answers_2'] = self._a3.tolist()
            for i, p in enumerate(predictions_2):
                if abs(p - self._a3[i]) < deviation_3:
                    count += 1
            _test_data['mse_2'] = float(np.mean((self._a3 - predictions_2) ** 2))
            _test_data['score_2'] = count / len(self._a3)
            _test_data['epochs'] = epochs
            _test_data['hidden_layer_count'] = _fit_data['hidden_layer_count']
            _test_data['hidden_neurons_count'] = _fit_data['hidden_neurons_count']
            filer.save_json(f"data_files/test_data/{_fit_data['network'].lower()}_test_data.txt",
                            _test_data)
            plotter.one_fit_statistic(_fit_data, _test_data)
