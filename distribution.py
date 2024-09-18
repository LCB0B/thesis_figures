import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import matplotlib.colors as mcolors

# set font to helvetica
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['helvetica']
plt.rcParams.update({'font.size': 12})

colors = ["#D55E00", "#FF9123", "#61C2FF", "#07AB92", "#014C4C", "#FF7DBE"]
main_color = '#DDDDDD'
second_color = colors[5]

cm = 1 / 2.54  # centimeters in inches

def darker(color, amount=0.5):
    """
    darken the given color.

    parameters:
    - color: str or tuple, color in any format matplotlib accepts.
    - amount: float, amount to darken (between 0 and 1).

    returns:
    - darker_color: tuple, rgb values of the darker color.
    """
    c = np.array(mcolors.to_rgb(color))
    return tuple(c * amount)

# create 'figures' directory if it doesn't exist
os.makedirs('figures', exist_ok=True)


def plot_heavy_tailed_distributions(filename='figures/heavy_tailed_distribution.png'):
    """
    generates data from exponential, lognormal, and power-law distributions,
    plots the data along with fitted lines.

    parameters:
    - filename_prefix: str, prefix for the filenames of the saved plots.
    """
    # generate data
    data_exponential = np.random.exponential(scale=0.5, size=1000)
    data_lognormal = np.random.lognormal(mean=0.0, sigma=2.0, size=1000)
    data_power_law = (np.random.pareto(a=2, size=1000) + 1) * 1

    # prepare data for plotting
    datasets = {
        'exponential': {
            'data': data_exponential,
            'color': colors[5],
        },
        'lognormal': {
            'data': data_lognormal,
            'color': colors[2],
        },
        'power law': {
            'data': data_power_law,
            'color': colors[3],
        },
    }

    plt.figure(figsize=(10 * cm, 7 * cm))
    maker_list = ['o', 's', '^']
    for dist_name, dist_info in datasets.items():
        data = dist_info['data']
        color = dist_info['color']

        # sort data
        data_sorted = np.sort(data)
        yvals = np.arange(1, len(data_sorted)+1) / float(len(data_sorted))

        # compute CCDF
        ccdf = 1 - yvals

        # plot empirical data
        #change markers for each distribution
        # plt.loglog(data_sorted, ccdf, marker='o', linestyle='None',
        #            markersize=4, color=color, markerfacecolor='none',
        #            markeredgewidth=1, label=f'{dist_name} data')
        plt.loglog(data_sorted, ccdf, marker=maker_list.pop(0), linestyle='None',
                    markersize=4, color=color, markerfacecolor='none',
                    markeredgewidth=1, label=f'{dist_name} data')
        lw = 1.5
        # fit the distribution
        if dist_name == 'exponential':
            # fit exponential distribution
            loc, scale = stats.expon.fit(data, floc=0)
            fitted_ccdf = 1 - stats.expon.cdf(data_sorted, loc=loc, scale=scale)
            plt.loglog(data_sorted, fitted_ccdf, linestyle='-', color=darker(color),
                       linewidth=lw, label=f'{dist_name} fit')
        elif dist_name == 'lognormal':
            # fit lognormal distribution
            shape, loc, scale = stats.lognorm.fit(data, floc=0)
            fitted_ccdf = 1 - stats.lognorm.cdf(data_sorted, shape, loc=loc, scale=scale)
            plt.loglog(data_sorted, fitted_ccdf, linestyle='-', color=darker(color),
                       linewidth=lw, label=f'{dist_name} fit')
        elif dist_name == 'power law':
            # fit power-law distribution using MLE for alpha
            xmin = data_sorted.min()
            alpha = 1 + len(data) / np.sum(np.log(data / xmin))
            fitted_ccdf = (data_sorted / xmin) ** (-alpha + 1)
            plt.loglog(data_sorted, fitted_ccdf, linestyle='-', color=darker(color),
                       linewidth=lw, label=f'{dist_name} fit')

    # set labels in lowercase
    plt.xlabel('x')
    plt.ylabel('ccdf')

    # remove the top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # adjust ticks and grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.2)
    from matplotlib.lines import Line2D
    # create custom legend handles for empirical data and fitted lines
    custom_lines = [
        Line2D([0], [0], marker='o', color=colors[5], markersize=5, markerfacecolor='none', label='exponential', linewidth=2, markeredgewidth=1),
        Line2D([0], [0], marker='s', color=colors[2], markersize=5, markerfacecolor='none', label='lognormal', linewidth=2, markeredgewidth=1),
        Line2D([0], [0], marker='^', color=colors[3], markersize=5, markerfacecolor='none', label='power law', linewidth=2, markeredgewidth=1),
    ]

    # set labels in lowercase
    plt.xlabel('x')
    plt.ylabel('ccdf')

    # remove the top and right spines
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # adjust ticks and grid
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.2)

    # create merged legend
    plt.legend(handles=custom_lines, loc='best', fontsize=10, frameon=False)


    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    
    
plot_heavy_tailed_distributions()
