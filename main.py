import typer
import modules

app = typer.Typer()


@app.command()
def main(cts: bool = typer.Option(False, help="Create, train and save model")):
    dataset = modules.DataSet("./data/data.csv", 900)
    X_train, y_train, X_test, y_test = dataset.load()

    model_helper = modules.KerasModel((X_test, y_test), (X_train, y_train))

    if cts:
        model = model_helper.create()
        trained_model = model_helper.train(model, 30)
        trained_model.save("./obj/v1.h5")

    model = model_helper.load("./obj/v1.h5")

    score = model_helper.test(model)
    print("Score:", score)


if __name__ == "__main__":
    app()