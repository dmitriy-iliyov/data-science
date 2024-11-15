class MyNetworkComparor:

    def __init__(self, custom_model, keras_model):
        self._custom_model = custom_model
        self._keras_model = keras_model

        self.km_fit_stat = None
        self.cm_fit_stat = None
        self.km_pred_stat = None
        self.cm_pred_stat = None

    def fit(self, train_data_set, epochs=100):
        train_data, train_answers = train_data_set
        self.cm_fit_stat = self._custom_model.fit(train_data, train_answers, epochs=epochs)
        self.km_fit_stat = self._keras_model.fit(train_data, train_answers, epochs=epochs)

        self.print_fit_plots()

    def predict(self, first_test_data_set, second_test_data_set):
        test_data_1, test_answers_1 = first_test_data_set
        prediction_c_1 = [self._custom_model.predict(i) for i in test_data_1]
        prediction_k_1 = self._keras_model.predict(test_data_1)

        test_data_2, test_answers_2 = second_test_data_set
        prediction_k_2 = self._keras_model.predict(test_data_2)
        prediction_c_2 = [self._custom_model.predict(i) for i in test_data_2]


    def print_fit_plots(self, ):
        CurrentPlotter = plotter.Plotter(14, 10, 2, 2, [1, 1], [1, 1])

        CurrentPlotter.add_fit_plots(self.cm_fit_stat, 0, 0, 0, 1)
        CurrentPlotter.add_fit_plots(self.km_fit_stat, 1, 0, 1, 1)
        CurrentPlotter.show()

    def prepared_data_to_plotting(self, pred_1, a_1, pred_2, a_2, model):
        model_pred_stat = {}
        count = 0
        deviation_2 = max(a_1) / 100
        model_pred_stat['test_predicts_1'] = pred_1
        model_pred_stat['test_answers_1'] = a_1.tolist()
        for i, p in enumerate(pred_1):
            if abs(p - a_1[i]) < deviation_2:
                count += 1
        model_pred_stat['mse_1'] = float(np.mean((a_1 - pred_1) ** 2))
        model_pred_stat['score_1'] = count / len(a_1)
        count = 0
        deviation_3 = max(a_2) / 100
        model_pred_stat['test_predicts_2'] = pred_2
        model_pred_stat['test_answers_2'] = a_2.tolist()
        for i, p in enumerate(pred_2):
            if abs(p - a_2) < deviation_3:
                count += 1
        model_pred_stat['mse_2'] = float(np.mean((a_2 - pred_2) ** 2))
        model_pred_stat['score_2'] = count / len(a_2)
        # model_pred_stat['epochs'] = epochs
        # model_pred_stat['hidden_layer_count'] = _fit_data['hidden_layer_count']
        # model_pred_stat['hidden_neurons_count'] = _fit_data['hidden_neurons_count']
