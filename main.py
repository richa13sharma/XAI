import modules

dataset = modules.DataSet("./data/data.csv", 900)
X_train, y_train, X_test, y_test = dataset.load()