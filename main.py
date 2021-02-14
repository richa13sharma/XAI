import typer
import modules
import perturb

app = typer.Typer()


@app.command()
def main(
    cts: bool = typer.Option(False, help="Create, train and save model"),
    precision_score: bool = typer.Option(False, help="Compute model score"),
    with_perturbation: bool = typer.Option(True, help="Compute model score"),
    model_path: str = typer.Argument("./obj/v1/h5", help="Path to saved model")
):

    dataset = modules.DataSet("./data/data.csv", 900)
    X_train, y_train, X_test, y_test = dataset.load()

    Model = modules.KerasModel((X_test, y_test), (X_train, y_train))

    if cts:
        Model.create()
        Model.train(epochs=30)
        Model.save(model_path)

    Model.load(model_path)

    if precision_score:
        print(f"Score: {Model.test()}")

    if with_perturbation:
        perturb.perturbateExperiment(Model)


if __name__ == "__main__":
    app()
