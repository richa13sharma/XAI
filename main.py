import modules

dataset = modules.DataSet("./data/data.csv", 900)
X_train, y_train, X_test, y_test = dataset.load()

model_helper = modules.KerasModel((X_test, y_test), (X_train, y_train))

"""
Uncomment the following lines to create, train and save model.
"""
# model = model_helper.create()
# trained_model = model_helper.train(model, 30)
# trained_model.save("./obj/v1.h5")

model = model_helper.load("./obj/v1.h5")

score = model_helper.test(model)
print("Score:", score)