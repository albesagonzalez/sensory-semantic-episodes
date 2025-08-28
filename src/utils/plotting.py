import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def plot_distribution(network, region, timestep, sleep_A=True, bins=50, get_x=False):
    if sleep_A == True:
        x = torch.stack(network.activity_recordings[region], axis=0)[network.sleep_indices_A][timestep]
    else:
        x = torch.stack(network.activity_recordings[region], axis=0)[timestep]

    plt.hist(x.flatten(), bins)
    plt.axhline(10)

    if get_x:
        return x


def plot_snapshot(network, region, timestep, sleep_A=True, get_x=False):
    if sleep_A == True:
        x = torch.stack(network.activity_recordings[region], axis=0)[network.sleep_indices_A][timestep]
    else:
        x = torch.stack(network.activity_recordings[region], axis=0)[timestep]

    plt.imshow(x.reshape((-1, 10)))

    if get_x:
        return x
    


from scipy.stats import sem
from scipy.stats import mannwhitneyu


# Define significance mapping
def p_to_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'




def compare_distributions_12(dist1, dist2, palette, size=(3.5, 3.5)):

    name1, list1 = dist1
    name2, list2 = dist2

    # Combine them for plotting
    data = np.concatenate([list1, list2])

    groups = ([name1] * len(list1) + 
            [name2] * len(list2))


    plt.figure(figsize=size)
    sns.violinplot(x=groups, y=data, inner=None, cut=0, alpha=0.9, palette=palette)
    sns.stripplot(x=groups, y=data, color='k', alpha=0.05, jitter=0.2, size=3)


    # Compute mean and SEM per group
    means = [np.mean(list1), np.mean(list2)]
    sems = [sem(list1), sem(list2)]

    # Overlay mean and error bars
    x_positions = range(2)
    plt.errorbar(x_positions, means, yerr=sems, fmt='o', color='black', capsize=5, markersize=6, lw=2, zorder=10)


    # Compute p-values
    p = mannwhitneyu(list1, list2, alternative='two-sided').pvalue


    y_max = max([m + s for m, s in zip(means, sems)]) + 0.08
    short_bar = 0.12
    tall_bar = 0.14
    bar_kwargs = dict(clip_on=False, color='black', linewidth=2)

    # Lower bar: (0-2)
    y1 = y_max
    plt.plot([0, 0, 1, 1], [y1, y1+short_bar, y1+short_bar, y1], **bar_kwargs)
    plt.text(0.5, y1+short_bar, p_to_star(p), ha='center', va='bottom', fontsize=16)



