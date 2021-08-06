import json
import os
from copy import copy
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# plot_data = {}
# metrics = []
#
# algos = []
# envs = []
#
# label = "qmix_4t_h1"
# path = "/home/pmatthaei/Projects/ma-league/results/league_2021-07-30_13-00-19/instance_0/sacred/1/info.json"
# out_path = "/home/pmatthaei/Projects/ma-league/results/league_2021-07-30_13-00-19/instance_0/sacred/1/" + "plots/"

def _to_value(x):
    return (x['value'] if 'value' in x else x['values']) if isinstance(x, dict) else x

def extract_plot_data(paths: List[str]) -> Dict:
    plot_data = {}
    xs_original = None
    for p in paths:
        # Read info json
        with open(p) as file:
            data = json.load(file)
            for metric in data:
                if '_T' not in metric:  # Skip, build by ourselves

                    # Get data for this metric and the current experiment
                    transposed_identifier = f"{metric}_T"
                    xs = data[transposed_identifier]
                    if xs_original is None:
                        xs_original = xs  # Used for unifying xs data
                    ys = data[metric]
                    ys = list(map(_to_value, ys))

                    if np.all(np.isnan(ys)):
                        continue
                    # Save the experiment data in the plot dict and provide label
                    if metric in plot_data:
                        plot_data[metric]["step"].extend(xs_original[:len(ys)])
                        plot_data[metric]["ys"].extend(ys)
                    else:
                        plot_data[metric] = {}
                        plot_data[metric].update({"step": xs, "ys": ys})

    return plot_data


def plot(data: Dict, out_path:str, show=False):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    for metric, metric_data in data.items():
        # Exclude for now since weird jsons are returned
        if metric in ['home_epsilon', 'grad_norm']:
            continue
        plt.figure()
        df = pd.DataFrame(metric_data)
        sns.lineplot(x="step", y="ys", ci=95, data=df, sort=True)
        plt.ylabel(metric.replace("_", " "))
        plt.xlim(left=-1)
        if "percentage" in metric:
            plt.ylim(bottom=0, top=1.0)
        plt.savefig(out_path + metric + '.png')
        if show:
            plt.show()
        plt.clf()

if __name__ == '__main__':
    name = "QMIX"
    path = "/home/pmatthaei/Projects/ma-league-results/saba/results/league_2021-08-05_19-02-10/instance_0/sacred/1"
    paths = [path.replace("instance_0", f"instance_{i}")for i in range(5)]
    for p in paths:
        data = extract_plot_data(paths=[f"{p}/info.json"])
        plot(data, out_path=f"{p}/plots/")
