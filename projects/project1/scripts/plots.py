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
                     'text.usetex': True,
                     'figure.titlesize': 22})

sns.set_style('whitegrid')
sns.set_palette(sns.color_palette('colorblind'))

COLOURS = list(sns.color_palette('colorblind'))

LABELSIZE = 18

def cross_validation_visualization(lambdas, losses_train, losses_test):
    """visualization the curves of mse_tr and mse_te."""

    avg_loss_tr = np.nanmean(losses_train, axis=1)
    std_loss_tr = np.nanstd(losses_train, axis=1)

    avg_loss_te = np.nanmean(losses_test, axis=1)
    std_loss_te = np.nanstd(losses_train, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xscale('log')

    ax.errorbar(lambdas, avg_loss_tr, yerr=std_loss_tr, marker="^", label='$\mathtt{Train}$')
    ax.errorbar(lambdas, avg_loss_te, yerr=std_loss_te, marker="o", label='$\mathtt{Test}$')

    ax.set(xlabel = "$\lambda$", ylabel='Misclassification rate')
    ax.legend(loc=2, fontsize=LABELSIZE)
    ax.grid(True)

    return fig


def model_comparison_visualization(accuracy, labels, acc_baseline):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.barh([i for i in range(len(accuracy))], accuracy)
    ax.vlines(acc_baseline, *ax.get_ylim(), colors=COLOURS[1], linestyle='--', linewidth=2.5)

    ax.set(xlabel='$\mathtt{Accuracy}$',
           yticks=[i for i in range(len(labels))], yticklabels=labels)

    return fig


def compare_features_visualisation(acc_train, acc_test, labels, model):
    arr_train = acc_train.flatten()
    arr_test = acc_test.flatten()

    arr_features = np.concatenate([np.repeat(labels, acc_train.shape[1]), np.repeat(labels, acc_train.shape[1])])
    arr_labels = np.concatenate([np.repeat('Train', len(arr_train)), np.repeat('Test', len(arr_test))])
    arr_acc = np.concatenate([arr_train, arr_test])

    fig, ax = plt.subplots(figsize=(8,3.5))
    ax = sns.boxplot(x=arr_features, y=arr_acc, hue=arr_labels)
    ax.set(ylabel='$\mathtt{Accuracy}$', title=f'{model}')

    return fig