#!/usr/bin/env python3
import copy
import math
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import imageio
import glob
import pdb
import modules
from sklearn.model_selection import KFold
from pathlib import Path


def perturbate(
    input_vars: np.array(np.array),
    column_number: int,
    value: float,
    sampling_method: str = "all",
    output_vars=None,
):
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

    features = copy.deepcopy(input_vars)
    outputs = copy.deepcopy(output_vars)

    # TODO: Switch based on an enum
    if sampling_method is "all":
        # Substitute all values of feature with `value`
        features[:, column_number] = [value] * len(features[:, column_number])

    elif sampling_method is "3fold":
        kf = KFold(n_splits=3, shuffle=True, random_state=1)
        kf.get_n_splits(features, outputs)
        for pert_index, unchanged_index in kf.split(input_vars, output_vars):
            Xu, Xp = features[pert_index], features[unchanged_index]
            yu, yp = outputs[pert_index], outputs[unchanged_index]
        # Perturbate 1st fold, leave other two as is
        Xp[:, column_number] = [value] * len(Xp[:, column_number])
        features = np.concatenate((Xp, Xu))
        outputs = np.concatenate((yp, yu))

    return (features, outputs)


def neuron_heatmap(relevance_scores, name, blue=False, title=None):
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
    fig.text(0.5, 0.25, str(name), ha="center")
    axes = axes.flatten()
    fig.suptitle(title)

    for i, layer in enumerate(relevance_scores):
        # pdb.set_trace()
        axes[i] = sns.heatmap(
            layer[:, np.newaxis],
            ax=axes[i],
            cbar=True,
            cmap=("coolwarm" if blue else None),
            # TODO: Fix 0 and trigger auto-scale
            vmin=-0.002,  # Use this have a consistent scale across layers
            vmax=0.002,
        )

    Path("./images").mkdir(exist_ok=True)
    fig.savefig("./images/" + name + ".jpg")
    plt.close()


def average_relevance(relevance_scores):
    """
    Computes average relevance scores across samples

    Args:
        relevance_scores: relevance matrix (layers * neurons)
    """
    return list(map(lambda layer: sum(layer) / len(layer), relevance_scores))


def _list_subtract(a, b):
    return np.array(a) - np.array(b)


def animate():
    filenames = glob.glob("./images/*.jpg")
    images = []
    filenames = sorted(filenames)
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave("./images/Age.gif", images, duration=1)


def _compute_inflection_point_helper(
    neuronRelevance, layer, pertubLevel, inflection_points
):
    for neuron in range(len(neuronRelevance)):
        if abs(neuronRelevance[neuron]) < abs(inflection_points[layer][neuron][1]):
            inflection_points[layer][neuron] = (
                pertubLevel + 1,
                neuronRelevance[neuron],
            )


def compute_inflection_point(avgRcollection):
    inflection_points = [
        [(0, math.inf)] * len(avgRcollection[0][0]),
        [(0, math.inf)] * len(avgRcollection[0][1]),
        [(0, math.inf)] * len(avgRcollection[0][2]),
    ]
    for pertubLevel in range(len(avgRcollection)):
        for layer, neuronRelevance in enumerate(avgRcollection[pertubLevel]):
            _compute_inflection_point_helper(
                neuronRelevance, layer, pertubLevel, inflection_points
            )

    return inflection_points


def binaryPerturbation(model):
    weights, bias = model.get_Weights_Bias()
    input_vars = model.X_train.values
    output_vars = model.y_train.values
    LRP_Helper = modules.LRPHelper(weights, bias, input_vars, output_vars)

    # Original (without perturbation)
    R = LRP_Helper.compute_LRP(input_vars=input_vars)
    avgR = average_relevance(R)
    neuron_heatmap(avgR, "0", title="Original")
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


def discretePerturbation(model):
    weights, bias = model.get_Weights_Bias()
    input_vars = model.X_train.values
    output_vars = model.y_train.values
    LRP_Helper = modules.LRPHelper(weights, bias, input_vars, output_vars)

    # Original (without perturbation)
    R = LRP_Helper.compute_LRP(input_vars=input_vars)
    avgR = average_relevance(R)
    avgRcollection = []
    step = 0.1
    n = 11
    for i in range(0, n):
        # Calculate the perturb value from 0 to 1 in this case
        val = step * i
        pert_input_vars, pert_output_vars = perturbate(
            input_vars, 1, val, output_vars=output_vars, sampling_method="3fold"
        )
        pertR = average_relevance(
            LRP_Helper.compute_LRP(
                input_vars=pert_input_vars, output_vars=pert_output_vars
            )
        )
        avgRcollection.append(pertR)
        # Difference heatmap
        neuron_heatmap(
            _list_subtract(avgR, pertR),
            "%.2d" % i,
            blue=True,
            title="Original-" + str("%.1f" % val),
        )
    animate()
    print(compute_inflection_point(avgRcollection))


def perturbateExperiment(model: modules.KerasModel):
    # binaryPerturbation(model)
    discretePerturbation(model)
