#!/usr/bin/env python3

"""
Neural backed decision tree
"""

import numpy as np
from collections import defaultdict

import modules
from perturb import perturbate, average_relevance, np_subtract


class NBDT:
    def __init__(self, model: modules.KerasModel):
        self.model = model
        self.input_vars = model.X_train.values
        self.output_vars = model.y_train.values
        weights, bias = model.get_Weights_Bias()
        self.LRP_Helper = modules.LRPHelper(
            weights, bias, self.input_vars, self.output_vars
        )
        self.relevance_per_step = defaultdict(
            lambda: None
        )  # Currently key is the feature, later we can do 'feature + step'.

    def generate_relevance_per_step(self, feature, step=0.1):
        """
        Generates relevance scores per step on perturbation of a particular feature

        Returns: array of shape (steps, layer, neuron)
        """

        # Initialize to empty list
        self.relevance_per_step[feature] = []

        # Calculate original relevance scores
        original_avgR = average_relevance(
            self.LRP_Helper.compute_LRP(
                input_vars=self.input_vars, output_vars=self.output_vars
            )
        )

        # Calculate relevance score for each step
        for value in np.arange(0, 1, step).tolist():
            # perturbate data
            pert_input_vars, pert_output_vars = perturbate(
                input_vars=self.input_vars,
                column_number=feature,
                value=value,
                output_vars=self.output_vars,
                sampling_method="3fold",
            )
            # calculate relevance
            avgR = average_relevance(
                self.LRP_Helper.compute_LRP(
                    input_vars=pert_input_vars, output_vars=pert_output_vars
                )
            )

            # append relevance score
            self.relevance_per_step[feature].append(avgR)

            # # Note: We can also store the difference from the original
            # self.relevance_per_step[feature].append(np_subtract(original_avgR, avgR))

        return self.relevance_per_step[feature]

    def get_step_relevance_for_neuron(self, feature, layer, neuron, step=0.1):
        """
        Get the relevance scores for a neuron by perturbation

        Returns: relevance scores for a neuron - array of shape (step,)
        """
        if self.relevance_per_step[feature] is None:
            self.generate_relevance_per_step(feature, step)
        return list(map(lambda x: x[layer][neuron], self.relevance_per_step[feature]))
