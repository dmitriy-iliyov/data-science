from lab_2_fce.lab_2.tools import plotter, data_processing as dp


class NetworkComparor:

    def __init__(self, first_model, second_model):
        self._first_model = first_model
        self._second_model = second_model

        self._s_fit_stat = None
        self._f_fit_stat = None
        self._f_test_stat = None
        self._s_test_stat = None

    def fit(self, train_data_set, epochs=100):
        train_data, train_answers = train_data_set
        self._f_fit_stat = self._first_model.fit(train_data, train_answers, epochs=epochs)
        self._s_fit_stat = self._second_model.fit(train_data, train_answers, epochs=epochs)

    def predict(self, first_test_data_set):
        test_data_1, test_answers_1 = first_test_data_set
        prediction_f = self._first_model.predict(test_data_1)
        prediction_s = self._second_model.predict(test_data_1)

        self._f_test_stat = dp.prepared_data_to_plotting(prediction_f, test_answers_1)
        self._s_test_stat = dp.prepared_data_to_plotting(prediction_s, test_answers_1)

    def print_plots(self):
        current_plotter = plotter.Plotter(14, 10, 2, 2, [1, 1], [1, 1])
        current_plotter.add_mse_plots(self._f_fit_stat, 0, 0)
        current_plotter.add_test_plots(self._f_test_stat, 0, 1)
        current_plotter.add_mse_plots(self._s_fit_stat, 1, 0)
        current_plotter.add_test_plots(self._s_test_stat, 1, 1)
        current_plotter.show()
