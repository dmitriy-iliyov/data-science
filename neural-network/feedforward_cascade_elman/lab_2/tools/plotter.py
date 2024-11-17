import matplotlib.pyplot as plt
import mplcursors


def one_plot(ax, x, y, y_max, label, color, title, x_label, y_label):
    ax.plot(x, y, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, y_max + y_max / 10)
    ax.legend()
    ax.grid(True)

    mplcursors.cursor(ax).connect("add", lambda sel: sel.annotation.set_text(f"{sel.target[1]:.4f}"))


def two_plot(ax, x1, y1, label1, color1, x2, y2, label2, color2, title, x_label, y_label):
    ax.plot(x1, y1, label=label1, color=color1, marker='o', markersize=2)
    ax.plot(x2, y2, label=label2, color=color2, marker='o', markersize=2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    y = y1 + y2
    min_y = min(y)
    max_y = max(y)
    ax.set_ylim(min_y - min_y / 10, max_y + max_y / 10)
    ax.legend()
    ax.grid(True)

    mplcursors.cursor(ax).connect("add", lambda sel: sel.annotation.set_text(f"{sel.target[1]:.4f}"))


def one_fit_statistic(fit_data, test_data):
    fig = plt.figure(figsize=(14, 10))
    # fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.2)
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.95, wspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0])
    one_plot(ax1, range(fit_data['epochs']), fit_data['accuracy'], 1, 'Accuracy', 'blue', 'Model Accuracy',
             'Epochs', 'Accuracy')
    hidden_layer_count = fit_data.get('hidden_layer_count', 1)
    test_1_info = (f'Network: {fit_data["network"]}\n'
                   f'Hidden layer count: {hidden_layer_count}\n'
                   f'Hidden Neurons: {fit_data["hidden_neurons_count"]}\n'
                   f'Execution Time: {fit_data["execution_time"]:.2f}s\n'
                   f'Batch Size: {fit_data["batch_size"]}')
    ax1.text(0.975, 0.05, test_1_info, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
             transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    ax2 = fig.add_subplot(gs[0, 1])
    one_plot(ax2, range(fit_data['epochs']), fit_data['mse'], max(fit_data['mse']), 'MSE', 'red', 'Model MSE',
             'Epochs', 'MSE')
    mse_info = f"MSE: {fit_data['mse'][-1]:.4f}"
    ax2.text(0.975, 0.05, mse_info, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
             transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    ax3 = fig.add_subplot(gs[1, 0])
    sorted_test_predicts_1 = sorted(test_data['test_predicts_1'], reverse=True)
    sorted_answers_1 = sorted(test_data['test_answers_1'], reverse=True)
    e_count_1 = [i for i in range(len(test_data['test_answers_1']))]
    two_plot(ax3, e_count_1, sorted_test_predicts_1, 'Predicts', 'red',
             e_count_1, sorted_answers_1, 'Answers', 'green',
             'Test on fit range', 'Examples Count', 'Answers')
    test_1_info = (f'MSE: {test_data['mse_1']:.4f}\n'
                   f'Test Accuracy: {test_data['score_1']}')
    ax3.text(0.975, 0.05, test_1_info, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
             transform=ax3.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    ax4 = fig.add_subplot(gs[1, 1])
    sorted_test_predicts_2 = sorted(test_data['test_predicts_2'], reverse=True)
    sorted_answers_2 = sorted(test_data['test_answers_2'], reverse=True)
    e_count_2 = [i for i in range(len(test_data['test_answers_2']))]
    two_plot(ax4, e_count_2, sorted_test_predicts_2, 'Predicts', 'red',
             e_count_2, sorted_answers_2, 'Answers', 'green',
             'Test on new range', 'Examples Count', 'Answers')
    test_2_info = (f'MSE: {test_data['mse_2']:.4f}\n'
                   f'Test Accuracy: {test_data['score_2']}')
    ax4.text(0.975, 0.05, test_2_info, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
             transform=ax4.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()


def results_plots(network_name, tdfr_t1, afr_t1, tdnr_t1, anr_t1, tdfr_t2, afr_t2, tdnr_t2, anr_t2):
    network_name = network_name.upper()
    fig = plt.figure(figsize=(14, 10))
    # fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.2)
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.95, wspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0])
    e_count_1 = [i for i in range(len(tdfr_t1))]
    two_plot(ax1, e_count_1, tdfr_t1, 'Predicts', 'red',
             e_count_1, afr_t1, 'Answers', 'green',
             f"Test {network_name} type 1 on fit range", 'Examples Count', 'Answers')
    ax2 = fig.add_subplot(gs[0, 1])
    e_count_2 = [i for i in range(len(tdnr_t1))]
    two_plot(ax2, e_count_2, tdnr_t1, 'Predicts', 'red',
             e_count_2, anr_t1, 'Answers', 'green',
             f"Test {network_name} type 1 on new range", 'Examples Count', 'Answers')
    ax3 = fig.add_subplot(gs[1, 0])
    e_count_3 = [i for i in range(len(tdfr_t2))]
    two_plot(ax3, e_count_3, tdfr_t2, 'Predicts', 'red',
             e_count_3, afr_t2, 'Answers', 'green',
             f"Test {network_name} type 2 on fit range", 'Examples Count', 'Answers')
    ax4 = fig.add_subplot(gs[1, 1])
    e_count_4 = [i for i in range(len(tdnr_t2))]
    two_plot(ax4, e_count_4, tdnr_t2, 'Predicts', 'red',
             e_count_4, anr_t2, 'Answers', 'green',
             f"Test {network_name} type 2 on new range", 'Examples Count', 'Answers')
    plt.show()


def print_plot_lab_3(data):
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], height_ratios=[1], wspace=0.2, hspace=0.2)
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.95, wspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0])
    one_plot(ax1, range(data['epochs']), data['accuracy'], 1, 'Accuracy', 'blue', 'Model Accuracy',
             'Epochs', 'Accuracy')
    test_1_info = (f'Network: FNN\n'
                   f'Hidden layer count: 1\n'
                   f'Hidden Neurons: 128\n'
                   f'Execution Time: {data["execution_time"]:.2f}s\n'
                   f'Batch Size: {data["batch_size"]}')
    ax1.text(0.975, 0.05, test_1_info, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
             transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    ax2 = fig.add_subplot(gs[0, 1])
    one_plot(ax2, range(data['epochs']), data['cce'], max(data['cce']),
             'CCE', 'red', 'Categorical Cross Entropy',
             'Epochs', 'Categorical Cross Entropy')
    cce_info = f"CCE: {data['cce'][-1]:.4f}"
    ax2.text(0.975, 0.05, cce_info, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
             transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()


class Plotter:

    def __init__(self, x, y, nrows, ncols, w, h):
        self.fig = plt.figure(figsize=(x, y))
        self.gs = self.fig.add_gridspec(nrows, ncols, width_ratios=w, height_ratios=h, wspace=0.2, hspace=0.2)
        plt.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.95, wspace=0.2)

    def add_one_plot(self, gx, gy, x, y, y_max, label, color, title, x_label, y_label):
        ax = self.fig.add_subplot(self.gs[gx, gy])
        one_plot(ax, x, y, y_max, label, color, title, x_label, y_label)

    def add_two_plots(self, gx, gy, x1, y1, label1, color1, x2, y2, label2, color2, title, x_label, y_label):
        ax = self.fig.add_subplot(self.gs[gx, gy])
        two_plot(ax, x1, y1, label1, color1, x2, y2, label2, color2, title, x_label, y_label)

    def add_fit_plots(self, fit_data, gx_1, gy_1, gx_2, gy_2):
        ax1 = self.fig.add_subplot(self.gs[gx_1, gy_1])
        one_plot(ax1, range(fit_data['epochs']), fit_data['accuracy'], 1, 'Accuracy', 'blue', 'Model Accuracy',
                 'Epochs', 'Accuracy')
        hidden_layer_count = fit_data.get('hidden_layer_count', 1)
        test_1_info = (f'Network: {fit_data["network"]}\n'
                       f'Hidden layer count: {hidden_layer_count}\n'
                       f'Hidden Neurons: {fit_data["hidden_neurons_count"]}\n'
                       f'Execution Time: {fit_data["execution_time"]:.2f}s\n'
                       f'Batch Size: {fit_data["batch_size"]}')
        ax1.text(0.975, 0.05, test_1_info, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
                 transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.5))

        ax2 = self.fig.add_subplot(self.gs[gx_2, gy_2])
        one_plot(ax2, range(fit_data['epochs']), fit_data['mse'], max(fit_data['mse']), 'MSE', 'red', 'Model MSE',
                 'Epochs', 'MSE')
        mse_info = f"MSE: {fit_data['mse'][-1]:.4f}"
        ax2.text(0.975, 0.05, mse_info, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
                 transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    def add_mse_plots(self, fit_data, gx, gy):
        ax = self.fig.add_subplot(self.gs[gx, gy])
        one_plot(ax, range(fit_data['epochs']), fit_data['mse'], max(fit_data['mse']), 'MSE', 'red', 'Model MSE',
                 'Epochs', 'MSE')
        mse_info = f"MSE: {fit_data['mse'][-1]:.4f}"
        hidden_layer_count = fit_data.get('hidden_layer_count', 1)
        test_1_info = (f'Network: {fit_data["network"]}\n'
                       f'Hidden layer count: {hidden_layer_count}\n'
                       f'Hidden Neurons: {fit_data["hidden_neurons_count"]}\n'
                       f'Execution Time: {fit_data["execution_time"]:.2f}s\n'
                       f'Batch Size: {fit_data["batch_size"]}\n'
                       f'MSE: {fit_data['mse'][-1]:.4f}')
        ax.text(0.975, 0.05, test_1_info, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
                 transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    def add_test_plots(self, test_data, gx, gy):
        ax3 = self.fig.add_subplot(self.gs[gx, gy])
        sorted_test_predicts_1 = sorted(test_data['test_predicts_1'], reverse=True)
        sorted_answers_1 = sorted(test_data['test_answers_1'], reverse=True)
        e_count_1 = [i for i in range(len(test_data['test_answers_1']))]
        two_plot(ax3, e_count_1, sorted_test_predicts_1, 'Predicts', 'red',
                 e_count_1, sorted_answers_1, 'Answers', 'green',
                 'Test on fit range', 'Examples Count', 'Answers')
        test_1_info = (f'MSE: {test_data['mse_1']:.4f}\n'
                       f'Test Accuracy: {test_data['score_1']}')
        ax3.text(0.975, 0.05, test_1_info, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
                 transform=ax3.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    def show(self):
        plt.show()
