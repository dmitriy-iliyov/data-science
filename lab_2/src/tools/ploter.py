import matplotlib.pyplot as plt
import numpy as np

color_names = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']


def one_plot(data, label):
    data = np.array(data)
    if data is not None:
        ids = [i for i in range(len(data))]
        plt.figure(figsize=(10, 5))
        plt.plot(ids, data, label=label, color='orange')
        plt.title(label)
        plt.ylabel('USD')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print('No data found')


def two_plots(data_1, label_1, data_2, label_2):
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    if data_1 is not None and data_2 is not None:
        df_1_ids = [i for i in range(len(data_1))]
        df_2_ids = [i for i in range(len(data_2))]
        plt.figure(figsize=(10, 5))
        plt.plot(df_1_ids, data_1, label=label_1, color='orange')
        plt.plot(df_2_ids, data_2, label=label_2, color='blue')
        plt.ylabel('USD')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()
    else:
        print('No data found')


def three_plots(data_set, labels):
    data_1 = np.array(data_set[0])
    data_2 = np.array(data_set[1])
    data_3 = np.array(data_set[2])
    if data_1 is not None and data_2 is not None and data_3 is not None:
        df_1_ids = [i for i in range(len(data_1))]
        df_2_ids = [i for i in range(len(data_2))]
        df_3_ids = [i for i in range(len(data_3))]
        plt.figure(figsize=(10, 5))
        plt.plot(df_1_ids, data_1, label=labels[0], color='orange')
        plt.plot(df_2_ids, data_2, label=labels[1], color='red')
        plt.plot(df_3_ids, data_3, label=labels[2], color='blue')
        plt.ylabel('USD')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()
    else:
        print('No data found')


def dots(data_set):
    for data in data_set:
        color = color_names[data_set.index(data)]
        for i in data:
            plt.scatter(i[0], i[1], color=color, marker='o', s=3)
            plt.xlabel('X')
            plt.ylabel('Y')
    plt.show()


def hist(data):
    data = np.array(data)
    if data is not None:
        plt.figure(figsize=(10, 5))
        plt.hist(data, bins='auto')
        plt.show()
    else:
        print('No data found')


def clean():
    plt.close('all')
