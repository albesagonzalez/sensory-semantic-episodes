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




def compare_distributions_12(dist1, dist2, palette, size=(3.5, 3.5), compare_points=False):

    name1, list1 = dist1
    name2, list2 = dist2
    
    # Accept torch tensors, numpy arrays, or python lists.
    if torch.is_tensor(list1):
        list1 = list1.detach().cpu().numpy()
    if torch.is_tensor(list2):
        list2 = list2.detach().cpu().numpy()
    list1 = np.asarray(list1).reshape(-1)
    list2 = np.asarray(list2).reshape(-1)

    # Combine them for plotting
    data = np.concatenate([list1, list2])

    groups = ([name1] * len(list1) + 
            [name2] * len(list2))

    if isinstance(palette, dict):
        color1 = palette[name1]
        color2 = palette[name2]
    else:
        color1, color2 = palette[0], palette[1]


    plt.figure(figsize=size)
    sns.violinplot(x=groups, y=data, inner=None, cut=0, alpha=0.9, palette=palette)
    sns.stripplot(x=groups, y=data, color='k', alpha=0.05, jitter=0.2, size=3)

    if compare_points:
        if len(list1) != len(list2):
            raise ValueError(
                "compare_points=True requires both distributions to have the same number of points."
            )

        for y1_point, y2_point in zip(list1, list2):
            plt.plot(
                [0, 1],
                [y1_point, y2_point],
                color='black',
                alpha=0.12,
                linewidth=0.8,
                zorder=3,
            )


    # Compute mean and SEM per group
    means = [np.mean(list1), np.mean(list2)]
    sems = [sem(list1), sem(list2)]

    # Overlay mean and error bars
    x_positions = [0, 1]
    plt.errorbar(x_positions, means, yerr=sems, fmt='o', color='black', capsize=5, markersize=6, lw=2, zorder=10)
    plt.axhline(means[0], color=color1, linestyle='--', linewidth=2, alpha=0.9, zorder=9)
    plt.axhline(means[1], color=color2, linestyle='--', linewidth=2, alpha=0.9, zorder=9)


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


def compare_distributions_123(dist1, dist2, dist3, palette, size=(4, 4)):

    name1, list1 = dist1
    name2, list2 = dist2
    name3, list3 = dist3

    if torch.is_tensor(list1):
        list1 = list1.detach().cpu().numpy()
    if torch.is_tensor(list2):
        list2 = list2.detach().cpu().numpy()
    if torch.is_tensor(list3):
        list3 = list3.detach().cpu().numpy()

    list1 = np.asarray(list1).reshape(-1)
    list2 = np.asarray(list2).reshape(-1)
    list3 = np.asarray(list3).reshape(-1)

    data = np.concatenate([list1, list2, list3])
    groups = (
        [name1] * len(list1) +
        [name2] * len(list2) +
        [name3] * len(list3)
    )

    plt.figure(figsize=size)
    sns.violinplot(x=groups, y=data, inner=None, cut=0, alpha=0.9, palette=palette)
    sns.stripplot(x=groups, y=data, color='k', alpha=0.1, jitter=0.2, size=3)

    means = [np.mean(list1), np.mean(list2), np.mean(list3)]
    sems = [sem(list1), sem(list2), sem(list3)]

    x_positions = range(3)
    plt.errorbar(x_positions, means, yerr=sems, fmt='o', color='black', capsize=5, markersize=6, lw=2, zorder=10)

    p_12 = mannwhitneyu(list1, list2, alternative='two-sided').pvalue
    p_13 = mannwhitneyu(list1, list3, alternative='two-sided').pvalue
    p_23 = mannwhitneyu(list2, list3, alternative='two-sided').pvalue

    y_max = max([m + s for m, s in zip(means, sems)]) + 0.08
    short_bar = 0.12
    tall_bar = 0.14
    bar_kwargs = dict(clip_on=False, color='black', linewidth=2)

    y1 = y_max
    plt.plot([0, 0, 2, 2], [y1, y1 + short_bar, y1 + short_bar, y1], **bar_kwargs)
    plt.text(1.0, y1 + short_bar, p_to_star(p_13), ha='center', va='bottom', fontsize=16)

    y2 = y1 + short_bar + tall_bar
    plt.plot([0, 0, 0.95, 0.95], [y2, y2 + short_bar, y2 + short_bar, y2], **bar_kwargs)
    plt.text(0.5, y2 + short_bar + 0.01, p_to_star(p_12), ha='center', va='bottom', fontsize=16)

    plt.plot([1.05, 1.05, 2, 2], [y2, y2 + short_bar, y2 + short_bar, y2], **bar_kwargs)
    plt.text(1.5, y2 + short_bar + 0.01, p_to_star(p_23), ha='center', va='bottom', fontsize=16)
