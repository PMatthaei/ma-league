import json
import matplotlib.pyplot as plt
import seaborn as sns

plot_data = {}
metrics = []

algos = []
envs = []

label = "qmix_4t_h1"
path = "/home/pmatthaei/Projects/ma-league-results/sacred/6/info.json"
out_path="/home/pmatthaei/Projects/ma-league-results/sacred/6/" + "plots/"

def extract_plot_data():
    # Read json
    with open(path) as json_file:
        data = json.load(json_file)
        for metric in data:
            # Transposed corresponds to xs values. We identify metrics by ys
            if '_T' not in metric:
                # If we have not already seen this metric in a json -> add to plots
                if metric not in metrics:
                    metrics.append(metric)

                # Get data for this metric and the current experiment
                transposed_identifier = metric + '_T'
                xs = data[transposed_identifier]
                ys = data[metric]

                # Save the experiment data in the plot dict and provide label
                if metric in plot_data:
                    plot_data[metric].append((xs, ys, label))
                else:
                    plot_data[metric] = []
                    plot_data[metric].append((xs, ys, label))


# Smoothing parameters
n = 30  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
not_smooth = ['epsilon', 'td_error_abs', 'loss']


def plot():
    for metric in metrics:
        # Exclude for now since weird jsons are returned
        if metric in ['home_epsilon', 'grad_norm']:
            continue

        for i, data in enumerate(plot_data[metric]):
            # If plot data is hidden in a dict property 'value' -> unpack
            ys = list(map(lambda x: (x['value'] if 'value' in x else x['values']) if isinstance(x, dict) else x, data[1]))
            xs = data[0]

            # Confidence interval if we have multiple values
            sns.lineplot(x=xs, y=ys, ci=95, label=data[2])

            plt.xlabel('steps')
            plt.xlim(right=2000000)  # Experiments end at 2 million steps
            metric_label = metric.replace('_', ' ')
            plt.ylabel(metric_label)

        plt.legend(loc='upper left')
        plt.savefig(out_path + metric + '.png')
        plt.show()


if __name__ == '__main__':
    extract_plot_data()
    plot()
