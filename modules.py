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

    def __init__(self, path, train_split):
        """
        `path` is the path to the dataset
        `train_split` is the number of samples in train data
        """
        self.path = path
        self.train_split = train_split

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
        size = self.train_split

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

        X_train, y_train, X_test, y_test = self._split(data)

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

    def load(self, path):
        """
        Loads the model at the given path.
        `path` is the path to the model
        Returns:
        `model` Keras model object
        """
        from keras.models import load_model

        # model not found errors handled by tensorflow
        model = load_model(path)

        return model

    def create(self):
        """
        Creates a Keras model defined in this function.
        Returns:
        `model` Keras model object
        """
        from keras.models import Sequential
        from keras.layers import Dense

        model = Sequential()
        model.add(Dense(units=24, activation="relu", input_dim=24))
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

        from keras import optimizers

        sgd = optimizers.SGD(lr=0.03, decay=0, momentum=0.9, nesterov=False)

        model.compile(
            loss="sparse_categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
        )

        return model

    def train(self, model, epochs, batch_size=128):
        """
        Trains a given Keras model for the given number of epochs and batch size
        `model` is a Keras model to train
        `epochs` are the number of epochs
        `batch_size` is the size of the batch per epoch. Defaults to 128.
        Returns:
        `model` a trained model
        """
        model.fit(
            self.X_train.values,
            self.y_train.values,
            validation_data=(self.X_test.values, self.y_test.values),
            epochs=epochs,
            batch_size=batch_size,
        )

        return model

    def test(self, model):
        """
        Tests a given Keras model and returns a precision score.
        `model` is a Keras model to test
        Returns:
        `score` the precision score
        """
        y_pred = model.predict_classes(self.X_test.values)
        y_val = self.y_test.values

        from sklearn.metrics import precision_score

        precision_score = precision_score(y_val, y_pred, average="weighted")

        return precision_score
