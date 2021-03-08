#!/usr/bin/env python3

"""
Neural backed decision tree
"""

import numpy as np
from collections import defaultdict

import modules
from perturb import perturbate, average_relevance, np_subtract


class NBDT:
    def __init__(self, model: modules.KerasModel, topK: int):
        self.model = model
        self.topK = topK
        self.input_vars = model.X_train.values
        self.output_vars = model.y_train.values
        weights, bias = model.get_Weights_Bias()
        self.LRP_Helper = modules.LRPHelper(
            weights, bias, self.input_vars, self.output_vars
        )
        self.relevance_per_step = defaultdict(
            lambda: None
        )  # Currently key is the feature, later we can do 'feature + step'.

    def _decompose_dt_feature(self, index):
        """
        Given a flat index from the DT, returns the layer and neuron (0 indexed)
        `index` is the flat index
        Returns:
        `layer` layer ID
        `neuron` neuron ID
        """
        layer = index // self.topK

        top = index % self.topK
        # [[19, 3], [6, 2], [5, 9]]
        neuron = self.topAvgScoreNodeIndex[layer][top]

        return layer, neuron

    def dump_rules(self, path="./obj/rules.csv"):
        """
        Outputs the rules for each node, each feature after perturbation based on details from the node into a CSV file
        `path` is the path to the CSV file
        """
        n_nodes = self.classifier.tree_.node_count
        children_left = self.classifier.tree_.children_left
        children_right = self.classifier.tree_.children_right
        dt_feature = self.classifier.tree_.feature
        threshold = self.classifier.tree_.threshold

        features = list(self.model.X_train.columns)

        with open(path, "w+") as f:
            print(
                "dt_node",
                "label",
                "layer_id",
                "neuron_id",
                "feature",
                "high_range",
                "low_range",
                sep=", ",
                file=f,
            )

            for i in range(n_nodes):
                if children_left[i] == -1 and children_right[i] == -1:
                    continue
                for j, feature in enumerate(features):
                    label = self.labels[dt_feature[i]]
                    layer, neuron = self._decompose_dt_feature(dt_feature[i])
                    # Get ranges - low and high
                    steps = self.get_step_relevance_for_neuron(
                        feature=j, layer=layer, neuron=neuron, step=0.001
                    )
                    low, high = self.generate_step_range_pairs(steps, threshold[i])
                    print(i, label, layer, neuron, feature, high, low, sep=", ", file=f)

    def _dump_paths(self, classifier, X_test, path="./obj/path.log"):
        """
        Logging function to output the path for all test samples to a log file. The path is the sequence of nodes a sample traverses and the corresponding threshold value.
        `classifier` is the sklearn tree classifier
        `X_test` is a list of samples for which the path is logged
        `path` is the path of the log file
        """
        feature = classifier.tree_.feature
        threshold = classifier.tree_.threshold

        with open(path, "w+") as f:
            for i, X in enumerate(X_test):
                node_indicator = classifier.decision_path(X_test)
                leaf_id = classifier.apply(X_test)

                sample_id = i
                # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
                node_index = node_indicator.indices[
                    node_indicator.indptr[sample_id] : node_indicator.indptr[
                        sample_id + 1
                    ]
                ]

                print("\nSample {id}:\n".format(id=sample_id), file=f)
                for node_id in node_index:
                    # continue to the next node if it is a leaf node
                    if leaf_id[sample_id] == node_id:
                        continue

                    # check if value of the split feature for sample 0 is below threshold
                    if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
                        threshold_sign = "<="
                    else:
                        threshold_sign = ">"

                    print(
                        "Node {node}: (X[{sample}, {feature}] = {value}) "
                        "{inequality} {threshold})".format(
                            node=node_id,
                            sample=sample_id,
                            feature=feature[node_id],
                            value=X_test[sample_id, feature[node_id]],
                            inequality=threshold_sign,
                            threshold=threshold[node_id],
                        ),
                        file=f,
                    )

    def _plot_DT(self, classifier, hidden_layers, path="./img/DT.png"):
        """
        Saves the visualization of the DT using graphviz.
        `classifier` is the sklearn tree classifier
        `hidden_layers` is the number of hidden layers
        `path` is the path at which the image will be saved
        """
        from sklearn.tree import export_graphviz
        from six import StringIO
        from IPython.display import Image
        import pydotplus

        labels = []
        for layer in range(hidden_layers):
            for top in range(self.topK):
                labels.append("Layer-" + str(layer + 1) + ":Top-" + str(top + 1))

        self.labels = labels

        dot_data = StringIO()
        export_graphviz(
            classifier,
            out_file=dot_data,
            filled=True,
            feature_names=labels,
            class_names=["0", "1"],
        )

        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        img = Image(graph.create_png())

        open(path, "wb+").write(img.data)

    def create_DT(self):
        """
        A public function to create the DT base on the R values.
        Returns:
        `accuracy` is the accuracy of the DT
        """
        features, self.topAvgScoreNodeIndex = self.LRP_Helper.create_DT_inputs(
            self.topK, self.input_vars, self.output_vars
        )
        labels = self.output_vars

        from sklearn import tree
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.1,
            random_state=7,
        )
        clf = tree.DecisionTreeClassifier()
        clf.fit(X=X_train, y=y_train)
        y_predict = clf.predict(X_test)

        count = 0
        for i, j in zip(y_predict, y_test):
            if i == j:
                count += 1

        accuracy = round(count / len(y_predict) * 100, 3)

        self.classifier = clf

        self._plot_DT(clf, 3)

        self._dump_paths(clf, X_test)

        return accuracy

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
