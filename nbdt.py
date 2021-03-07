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

    def _generate_step_range_pairs_helper(self, per_step_nueron_relevance, predicate):

        step_range_pairs = []
        r_start = 0
        r_end = 0
        for i in range(len(per_step_nueron_relevance)):
            if predicate(per_step_nueron_relevance[r_end]):
                r_end += 1
            else:
                if r_end != r_start:
                    step_range_pairs.append(((r_start + 1) / 10, (r_end) / 10))
                r_end += 1
                r_start = r_end

        if predicate(per_step_nueron_relevance[-1]):
            step_range_pairs.append(((r_start + 1) / 10, (r_end) / 10))

        return step_range_pairs

    def generate_step_range_pairs(self, per_step_nueron_relevance, dt_node_relevance):
        """
        Get the pertubation step range pairs depeding on the relevance threshold of the
        DT node.

        Returns: A tuple containing two lists :
            1. The first contains step range pairs for which relevance is less than or equal to the threshold.
            2. The second contains step range pairs for which relevance is greater than the threshold.
        """
        return (
            self._generate_step_range_pairs_helper(
                per_step_nueron_relevance, lambda x: x <= dt_node_relevance
            ),
            self._generate_step_range_pairs_helper(
                per_step_nueron_relevance, lambda x: x > dt_node_relevance
            ),
        )
