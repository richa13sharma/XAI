import numpy as np
import pandas as pd

import utils


class DataSet:
    """
    Loads a dataset given a csv file. Returns X and y for test and train of given dataset file.
    Currently supports categorical datasets only.
    Functions:
    `load` uses the csv file at given path to construct dataframes for test and train
    """

    def __init__(self, path, train_split_percentage, num_PCs=False, auto_pca=True):
        """
        `path` is the path to the dataset
        `train_split` is the number of samples in train data
        `dims` is the number of dimensions the dataset should have after transformation
        """
        self.path = path
        self.train_split_percentage = train_split_percentage
        self.num_PCs = num_PCs
        self.do_reduction = bool(num_PCs)
        self.auto_pca = auto_pca
        print("[INFO] do_reduction:", self.do_reduction)

    def _normalize(self, column):
        """
        Function to normalize values in a column.
        `column` is the column to be normalized
        Returns:
        `result` is the modified column
        """
        result = column.copy()
        max_value = column.max()
        min_value = column.min()
        result = (column - min_value) / (max_value - min_value)

        return result

    def _split(self, data):
        """
        Splits the processed data into test and train arrays.
        `data` is the dataframe to be split
        Returns:
        `X_train` is X_train
        `y_train` is y_train
        `X_test` is X_test
        `y_test` is y_test
        """
        size = int(self.train_split_percentage * len(data))

        # TODO: Add a function to shuffle data

        data_train = data.iloc[:size]
        data_valid = data.iloc[size:]

        X_train = data_train.iloc[:, :-1]
        y_train = data_train.iloc[:, -1]
        X_test = data_valid.iloc[:, :-1]
        y_test = data_valid.iloc[:, -1]

        return X_train, y_train, X_test, y_test

    def load(self):
        """
        Loads the dataset by returning X and y for test and train.
        Returns:
        `X_train` is X_train
        `y_train` is y_train
        `X_test` is X_test
        `y_test` is y_test
        """
        data = pd.read_csv(self.path, index_col=0, sep=",")
        labels = data.columns

        from pandas.api.types import is_string_dtype

        for col in labels:
            if is_string_dtype(data[col]):
                if col == "Risk":
                    # convert risk to a binary value
                    data[col] = pd.factorize(data[col])[0]
                    continue

                # one-hot encoding
                data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)
                data.drop(col, axis=1, inplace=True)
            else:
                data[col] = self._normalize(data[col])

        # move 'Risk' back to the end of the df
        data = data[[c for c in data if c not in ["Risk"]] + ["Risk"]]

        # Transformations for dimensional reduction
        if self.do_reduction:
            transformed_data = utils.svd(
                data.drop(["Risk"], axis=1), components=self.num_pcs
            )
            final_df = pd.concat([transformed_data, data["Risk"]], axis=1)
        elif self.auto_pca:
            transformed_data, mapping = utils.auto_pca(data.drop(["Risk"], axis=1))
            # mapppings are eigen vectors corresponding to each PC.
            final_df = pd.concat([transformed_data, data["Risk"]], axis=1)
        else:
            final_df = data

        X_train, y_train, X_test, y_test = self._split(final_df)
        self.dims = len(final_df.columns) - 1  # subtract 1 to remove output column
        return X_train, y_train, X_test, y_test


class KerasModel:
    """
    On input of a dataset (test and train), loads or creates a model, trains and evaluates it.
    Functions:
    `load` loads an existing model
    `create` creates a model
    `train` trains a model
    `test` evaluates a model
    """

    def __init__(self, test, train):
        """
        `test` is a tuple of test X and y
        `train` is a tuple of train X and y
        """
        self.X_test = test[0]
        self.y_test = test[1]
        self.X_train = train[0]
        self.y_train = train[1]
        self.model = None

    def load(self, path):
        """
        Loads the model at the given path.
        `path` is the path to the model
        Returns:
        `model` Keras model object
        """
        from keras.models import load_model

        # model not found errors handled by tensorflow
        self.model = load_model(path)
        return self.model

    def create(self, input_dims):
        """
        Creates a Keras model defined in this function.
        `input_dims` is the number of input columns
        Returns:
        `model` Keras model object
        """
        from keras.models import Sequential
        from keras.layers import Dense
        from keras import optimizers

        model = Sequential()
        model.add(Dense(units=24, activation="relu", input_dim=input_dims))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=16, activation="relu"))
        model.add(
            Dense(
                units=2,
                activation="softmax",
                kernel_initializer="glorot_normal",
                bias_initializer="zeros",
            )
        )

        sgd = optimizers.SGD(lr=0.03, decay=0, momentum=0.9, nesterov=False)

        model.compile(
            loss="sparse_categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
        )

        self.model = model
        return self.model

    def train(self, epochs, batch_size=128):
        """
        Trains a given Keras model for the given number of epochs and batch size
        `model` is a Keras model to train
        `epochs` are the number of epochs
        `batch_size` is the size of the batch per epoch. Defaults to 128.
        Returns:
        `model` a trained model
        """
        self.model.fit(
            self.X_train.values,
            self.y_train.values,
            validation_data=(self.X_test.values, self.y_test.values),
            epochs=epochs,
            batch_size=batch_size,
        )
        return self.model

    def test(self):
        """
        Tests a given Keras model and returns a precision score.
        `model` is a Keras model to test
        Returns:
        `score` the precision score
        """
        y_pred = np.argmax(self.model.predict(self.X_test.values), axis=-1)
        y_val = self.y_test.values

        from sklearn.metrics import precision_score

        precision_score = precision_score(y_val, y_pred, average="weighted")

        return precision_score

    def save(self, path):
        """
        Function to save a Keras model of the class
        `path` to store the model at
        """
        self.model.save(path)

    def get_Weights_Bias(self):
        """
        Retrieves the weights and bias for a given keras model.
        `model` is a Keras model to retrieve weights of.
        Returns:
        `W` is Weights
        `B` is Bias
        """
        W = []
        B = []

        for layer in self.model.layers:
            W.append(layer.get_weights()[0])
            B.append(layer.get_weights()[1])

        W = np.array(W, dtype=object)
        B = np.array(B, dtype=object)

        return (W, B)


class LRPHelper:
    """
    On input of model weights, bias, training values computes the LRP values.
    Functions:
    `computeLRP` compute LRP values for a given model.
    """

    def __init__(self, W, B, x_train, y_train):
        """
        `W` is a np array of weights of the model
        `B` is a np array of bais of the model
        'X_train' is the training input values
        'y_train' is the training output values
        """

        self.W = W
        self.B = B
        self.x_train = x_train
        self.y_train = y_train

    def _rho(self, w, l):
        """helper functions that perform the weight transformation"""
        return w + [None, 0.1, 0.0, 0.0][l] * np.maximum(0, w)

    def _incr(self, z, l):
        """helper functions that perform the _incrementation"""
        return z + [None, 0.0, 0.1, 0.0][l] * (z ** 2).mean() ** 0.5 + 1e-9

    def _compute_model_output(self, X, W, B, L, printit=False):
        """
        Computes the output values of the model given the
        weights and bias
         X' is the training input values
        `W` is a np array of weights of the model
        `B` is a np array of bais of the model
        'L' is the length of W
        Returns:
        `A` the model output values
        """
        A = [X] + [None] * L

        for l in range(L):
            comp_val = A[l].dot(W[l]) + B[l]
            A[l + 1] = np.maximum((0.01) * comp_val, comp_val)

        if printit:
            for i in range(5):
                p = A[L][i]
                print("  ".join(["[%1d] %.1f" % (d, p[d]) for d in range(2)]))

        return A

    def _remove_zero_relevance_rows(self, R):
        """
        Remove the rows in R which have zero relevance for all nuerons
        `R` is Matrix containing the relevance values of the nuerons
        Returns:
        `new_R` updates R with no rows with only zero values.
        """
        new_R = []

        for layer in R:
            newLayer = []
            zero = [0] * layer[0]
            for row in layer:
                if not np.array_equal(row, zero):
                    newLayer.append(row)
            new_R.append(newLayer)
        for i in new_R:
            if len(i) != len(new_R[0]):
                print("wrong")

        new_R = new_R[1 : len(new_R) - 1]
        return new_R

    def compute_LRP(self, input_vars=None, output_vars=None):
        """
        Computes the LRP values for the model for all
        ip rows of the dataset
        Returns:
        `R` is Matrix containing the relevance values of the nuerons
        """

        X = self.x_train if input_vars is None else input_vars
        T = self.y_train if output_vars is None else output_vars
        W = self.W
        B = self.B
        L = len(W)

        A = self._compute_model_output(X, W, B, L)
        R = [None] * L + [A[L] * (T[:, None] == np.arange(2))]

        print(len(R), L)

        for l in range(1, L)[::-1]:

            w = self._rho(W[l], l)
            b = self._rho(B[l], l)

            z = self._incr(A[l].dot(w) + b, l)  # step 1
            s = R[l + 1] / z  # step 2
            c = s.dot(w.T)  # step 3
            R[l] = A[l] * c  # step 4

        w = W[0]
        wp = np.maximum(0, w)
        wm = np.minimum(0, w)
        lb = A[0] * 0 - 1
        hb = A[0] * 0 + 1

        z = A[0].dot(w) - lb.dot(wp) - hb.dot(wm) + 1e-9  # step 1
        s = R[1] / z  # step 2
        c, cp, cm = s.dot(w.T), s.dot(wp.T), s.dot(wm.T)  # step 3
        R[0] = A[0] * c - lb * cp - hb * cm

        R = self._remove_zero_relevance_rows(R)

        return R

    def _get_top_k_average_neurons(self, newR, k, validInputLength):

        topAvgScoreNodeIndex = []
        for layer in newR:
            avg = [0] * len(layer[0])
            for row in layer:
                avg = [avg[i] + row[i] for i in range(len(row))]
            avg = [((avg[i] / validInputLength), i) for i in range(len(avg))]
            avg.sort(key=lambda x: x[0], reverse=True)
            index = [x[1] for x in (avg[:k])]
            topAvgScoreNodeIndex.append(index)

        return topAvgScoreNodeIndex

    def create_DT_inputs(self, k, input_vars=None, output_vars=None):
        R = self.compute_LRP(input_vars, output_vars)
        number_of_samples = len(R[0])

        topAvgScoreNodeIndex = self._get_top_k_average_neurons(R, k, number_of_samples)
        features = [[] for _ in range(number_of_samples)]

        for i in range(number_of_samples):
            array = []
            for j, elem in enumerate(topAvgScoreNodeIndex):
                for featureNo in range(k):
                    array.append(R[j][i][elem[featureNo]])
            features[i] = np.array(array)

        return np.array(features), topAvgScoreNodeIndex
