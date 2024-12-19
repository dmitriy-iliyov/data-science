import numpy as np
import pandas as pd


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def data_preparing(name, sample_data):
    values = sample_data[name]
    n_sample_data = int(len(values))
    S_real = np.zeros((n_sample_data))
    for i in range(n_sample_data):
        S_real[i] = values[i]
    return S_real


def matrix_generation(file_name):
    sample_data = pd.read_csv(file_name)
    line_sample_data = int(sample_data.shape[0])
    column_sample_data = int(sample_data.shape[1])
    line_column_matrix = np.zeros((line_sample_data, column_sample_data - 1))
    Title_sample_data = sample_data.columns
    for i in range(1, column_sample_data):
        column_matrix = data_preparing(Title_sample_data[i], sample_data)
        for j in range(len(column_matrix)):
            line_column_matrix[j, (i-1)] = column_matrix[j]
    return line_column_matrix


def matrix_adapter(line_column_matrix, line):
    column_sample_matrix = np.shape(line_column_matrix)
    line_matrix = np.zeros((column_sample_matrix[1]))
    for j in range(column_sample_matrix[1]):
        line_matrix[j] = line_column_matrix[line, j]
    return line_matrix


def Voronin(file_name, g_list, max_index_list):
    line_column_matrix = matrix_generation(file_name)
    column_matrix = np.shape(line_column_matrix)
    print(column_matrix)
    Integro = np.zeros((column_matrix[1]))

    f_list = [matrix_adapter(line_column_matrix, i) for i in range(len(line_column_matrix))]

    f_norm_list = [np.zeros((column_matrix[1])) for _ in range(len(g_list))]

    g_norm = sum(g_list)

    g_norm_list = [0] * len(g_list)
    for i in range(len(g_list)):
        g_norm_list[i] = g_list[i] / g_norm

    sum_f_list = [0] * len(g_list)
    for i in range(column_matrix[1]):
        for j in range(len(g_list)):
            if j + 1 in max_index_list:
                sum_f_list[j] = sum_f_list[j] + (1/f_list[j][i])
            else:
                sum_f_list[j] = sum_f_list[j] + f_list[j][i]

    for i in range(column_matrix[1]):
        for j in range(len(f_norm_list)):
            if j + 1 in max_index_list:
                f_norm_list[j][i] = (1/f_list[j][i]) / sum_f_list[j]
            else:
                f_norm_list[j][i] = f_list[j][i] / sum_f_list[j]

    for i in range(column_matrix[1]):
        for gi, fi in zip(g_norm_list, f_norm_list):
            Integro[i] = Integro[i] + (gi * (1 - fi[i]) ** (-1))
    print(f_norm_list)

    min_ = 10000
    opt = 0
    for i in range(column_matrix[1]):
        if min_ > Integro[i]:
            min_ = Integro[i]
            opt = i
    print('Інтегрована оцінка (scor):')
    print(Integro)
    print('Номер оптимального обчислювального комплексу:', opt + 1)
    return


Voronin('files/data.csv', [4, 4, 3, 2, 1, 1, 4, 4, 3], [1, 2, 3])
