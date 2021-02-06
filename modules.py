import numpy as np
import pandas as pd

import utils


class DataSet:
    """
    Loads a dataset given a csv file. Returns X and y for test and train of given dataset file.
    Currently supports categorical datasets only.
    """

    def __init__(self, path, train_split):
        """
        `path` is the path to the dataset
        `train_split` is the number of samples in train data
        Functions:
        `load` uses the csv file at given path to construct dataframes for test and train
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

        # TODO: Add a function to shuffle data

        return X_train, y_train, X_test, y_test
