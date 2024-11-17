import numpy as np
from collections import defaultdict

import feedforward_cascade_elman.lab_2.tools.filer as filer


def function(x, y):
    return np.sqrt(np.power(x, 2) + np.power(y, 2))


def x_2(x):
    return pow(x, 2)


def prepared_data_one_parm_func(min_=-1, max_=1, n=100, split=True, func=x_2):
    x = np.random.uniform(min_, max_, n)
    y = func(x)
    indices = np.random.permutation(n)
    x, y = x[indices].astype(np.float32), y[indices].astype(np.float32)
    if split:
        b = int(n/5)
        return x[:-b], y[:-b], x[-b:], y[-b:]
    return x, y


def prepared_data(min_=-1, max_=1, n=100, split=True, func=function):
    x = np.random.uniform(min_, max_, n)
    y = np.random.uniform(min_, max_, n)
    z = func(x, y)

    indices = np.random.permutation(n)
    x, y, z = x[indices], y[indices], z[indices]
    xy = np.array([[x[i], y[i]] for i in range(n)], dtype=np.float32)
    if split:
        b = int(n/5)
        return xy[:-b], z.astype(np.float32)[:-b], xy[-b:], z.astype(np.float32)[-b:]
    return xy, z.astype(np.float32)


def prepared_data_to_plotting(pred_1, a_1):
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
    return model_pred_stat


def foo(_type, epochs, sample_len):
    data = filer.read_from_dir('data_files/test_data')
    output_data = defaultdict(lambda: {
        'test_predicts_1': np.zeros(0),
        'test_predicts_1_count': 0,
        'test_answers_1': np.zeros(0),
        'test_answers_1_count': 0,
        'test_predicts_2': np.zeros(0),
        'test_predicts_2_count': 0,
        'test_answers_2': np.zeros(0),
        'test_answers_2_count': 0,
        'hidden_layer_count': None,
        'hidden_neurons_count': None
    })

    for file_name, statistics_list in data.items():
        for statistics in statistics_list:
            key = file_name[:3]
            print(statistics['epochs'], len(statistics['test_predicts_1']))
            if statistics['epochs'] == epochs and len(statistics['test_predicts_1']) == sample_len:
                if _type == 1:
                    process_statistics_type_1(key, statistics, output_data)
                elif _type == 2:
                    process_statistics_type_2(key, statistics, output_data)
            else:
                continue

    normalize_output_data(output_data)
    pre_result = {
        'tp1': {network: output_data[network]['test_predicts_1'].tolist() for network in output_data},
        'ta1': {network: output_data[network]['test_answers_1'].tolist() for network in output_data},
        'tp2': {network: output_data[network]['test_predicts_2'].tolist() for network in output_data},
        'ta2': {network: output_data[network]['test_answers_2'].tolist() for network in output_data},
    }

    result = {}

    for network in output_data.keys():
        result[network] = {
            'tp1': pre_result['tp1'][network],
            'ta1': pre_result['ta1'][network],
            'tp2': pre_result['tp2'][network],
            'ta2': pre_result['ta2'][network],
        }

    return result, [output_data[n]['test_predicts_1_count'] for n in output_data.keys()]


def process_statistics_type_1(key, statistics, output_data):
    if key == 'fnn' and statistics['hidden_layer_count'] == 1 and statistics['hidden_neurons_count'] == 10:
        foo_part(key, statistics, output_data)
    elif key == 'cnn' and statistics['hidden_layer_count'] == 1 and statistics['hidden_neurons_count'] == 20:
        foo_part(key, statistics, output_data)
    elif key == 'enn' and statistics['hidden_layer_count'] == 1 and statistics['hidden_neurons_count'] == 15:
        foo_part(key, statistics, output_data)


def process_statistics_type_2(key, statistics, output_data):
    if key == 'fnn' and statistics['hidden_layer_count'] == 1 and statistics['hidden_neurons_count'] == 20:
        foo_part(key, statistics, output_data)
    elif key == 'cnn' and statistics['hidden_layer_count'] == 2 and statistics['hidden_neurons_count'] == 10:
        foo_part(key, statistics, output_data)
    elif key == 'enn' and statistics['hidden_layer_count'] == 3 and statistics['hidden_neurons_count'] == 5:
        foo_part(key, statistics, output_data)


def normalize_output_data(output_data):
    for network, stats in output_data.items():
        for key in ['test_predicts_1', 'test_answers_1', 'test_predicts_2', 'test_answers_2']:
            count_key = key + '_count'
            if stats[count_key] > 0:
                stats[key] /= stats[count_key]


def foo_part(key, statistics, output_data):
    if output_data[key]['test_predicts_1'].size == 0:
        output_data[key]['test_predicts_1'] = np.zeros(len(statistics['test_predicts_1']))
        output_data[key]['test_answers_1'] = np.zeros(len(statistics['test_answers_1']))
        output_data[key]['test_predicts_2'] = np.zeros(len(statistics['test_predicts_2']))
        output_data[key]['test_answers_2'] = np.zeros(len(statistics['test_answers_2']))

    output_data[key]['test_predicts_1'] += statistics['test_predicts_1']
    output_data[key]['test_predicts_1_count'] += 1

    output_data[key]['test_answers_1'] += statistics['test_answers_1']
    output_data[key]['test_answers_1_count'] += 1

    output_data[key]['test_predicts_2'] += statistics['test_predicts_2']
    output_data[key]['test_predicts_2_count'] += 1

    output_data[key]['test_answers_2'] += statistics['test_answers_2']
    output_data[key]['test_answers_2_count'] += 1

