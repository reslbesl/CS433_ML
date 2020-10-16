import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'axes.spines.right': True,
                     'axes.spines.top': True,
                     'axes.edgecolor': 'k',
                     'axes.labelsize':20,
                     'axes.titlesize':22,
                     'xtick.labelsize': 18,
                     'xtick.color': 'k',
                     'ytick.labelsize': 18,
                     'ytick.color': 'k',
                     'grid.color':'0.7',
                     'font.family': 'serif',
                     'font.sans-serif': 'cm',
                     'text.usetex': False,
                     'figure.titlesize': 22})

sns.set_style('whitegrid')
sns.set_palette(sns.color_palette('colorblind'))

COLOURS = list(sns.color_palette('colorblind'))

def cross_validation_visualization(lambdas, losses_train, losses_test):
    """visualization the curves of mse_tr and mse_te."""

    avg_loss_tr = np.nanmean(losses_train, axis=1)
    std_loss_tr = np.nanstd(losses_train, axis=1)

    avg_loss_te = np.nanmean(losses_test, axis=1)
    std_loss_te = np.nanstd(losses_train, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale('log')

    ax.errorbar(lambdas, avg_loss_tr, yerr=std_loss_tr, marker="o", label='Train error')
    ax.errorbar(lambdas, avg_loss_te, yerr=std_loss_te, marker="o", label='Test error')

    ax.set(xlabel = "Lambda", ylabel='Error')
    ax.legend(loc=2)
    ax.grid(True)

    return fig
