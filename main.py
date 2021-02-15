import typer
import modules
import perturb

app = typer.Typer()


@app.command()
def main(
    model_path: str = typer.Option("./obj/v1.h5", help="Path to saved model.", metavar="PATH"),
    cts: bool = typer.Option(False, help="Create, train and save model."),
    precision_score: bool = typer.Option(False, help="Compute model score."),
    perturbate: bool = typer.Option(True, help="Perform perturbation experiment."),
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

    if perturbate:
        perturb.perturbateExperiment(Model)


if __name__ == "__main__":
    app()
