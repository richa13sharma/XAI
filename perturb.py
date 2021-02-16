#!/usr/bin/env python3
import copy

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

import modules


def perturbate(input_vars: np.array(np.array), column_number: int, value: float):
    """
    Purturbate data.

    Args:
        input_vars : Dataset samples
        column_number : Feature to perturbatee
        value : Value to fill

    Options:
        sampling method : all, 5 fold etc (TODO)

    Returns:
        A copy of `input_vars` with perturbation

    """

    samples = copy.deepcopy(input_vars)

    # Substitute all values of feature with `value`
    samples[:, column_number] = [value] * len(samples[:, column_number])
    return samples


def neuron_heatmap(relevance_scores, blue=False, title=None):
    """
    Creates a heatmap with relevance scores for a sample

    Use `plt.show()` to display figures

    Args:
        relevance_scores: relevance matrix (layers * neurons)
        blue: use coolwarm color
    """
    fig, axes = plt.subplots(ncols=len(relevance_scores), squeeze=False)
    # fig, axes = plt.subplots(ncols=len(relevance_scores))

    fig.subplots_adjust(wspace=1.5)
    axes = axes.flatten()
    fig.suptitle(title)

    for i, layer in enumerate(relevance_scores):
        axes[i] = sns.heatmap(
            layer[:, np.newaxis],
            ax=axes[i],
            cbar=True,
            cmap=("coolwarm" if blue else None),
            # TODO: Fix 0 and trigger auto-scale
            vmin=-0.1,  # Use this have a consistent scale across layers
            vmax=0.1,
        )


def average_relevance(relevance_scores):
    """
    Computes average relevance scores across samples

    Args:
        relevance_scores: relevance matrix (layers * neurons)
    """
    return list(map(lambda layer: sum(layer) / len(layer), relevance_scores))


def _list_subtract(a, b):
    return np.array(a) - np.array(b)


def perturbateExperiment(
    model: modules.KerasModel, save: bool, path: str = "image.png"
):

    weights, bias = model.get_Weights_Bias()
    input_vars = model.X_train.values
    output_vars = model.y_train.values
    LRP_Helper = modules.LRPHelper(weights, bias, input_vars, output_vars)

    # Original (without perturbation)
    R = LRP_Helper.compute_LRP(input_vars=input_vars)
    avgR = average_relevance(R)
    # neuron_heatmap(avgR)

    # Binary perturbation : column = 1, value = 0
    pert_input_vars = perturbate(input_vars, 1, 0)
    lowR = average_relevance(LRP_Helper.compute_LRP(input_vars=pert_input_vars))
    # neuron_heatmap(lowR)

    # Binary perturbation : column = 1, value = 1
    pert_input_vars = perturbate(input_vars, 1, 1)
    highR = average_relevance(LRP_Helper.compute_LRP(input_vars=pert_input_vars))
    # neuron_heatmap(highR)

    # Difference heatmap
    neuron_heatmap(_list_subtract(avgR, lowR), blue=True, title="Original-Low")
    neuron_heatmap(_list_subtract(avgR, highR), blue=True, title="Original-High")

    if save:
        plt.savefig(path)
    else:
        plt.show()
