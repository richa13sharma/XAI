import typer
import modules
import perturb
import utils

app = typer.Typer()


@app.command()
def main(
    create: bool = typer.Option(False, help="Create, train and save model"),
    score: bool = typer.Option(False, help="Compute model score"),
    with_pert: bool = typer.Option(True, help="Compute model score"),
    model_path: str = typer.Argument("./obj/v1/h5", help="Path to saved model"),
    dataset_dims: int = typer.Argument(24, help="Number of input dimensions"),
):

    dataset = modules.DataSet("./data/data.csv", 900, dataset_dims)
    X_train, y_train, X_test, y_test = dataset.load()

    Model = modules.KerasModel((X_test, y_test), (X_train, y_train))

    if create:
        Model.create(dataset_dims)
        Model.train(epochs=30)
        Model.save(model_path)

    Model.load(model_path)

    if score:
        print("Score: ", Model.test())

    if with_pert:
        perturb.perturbateExperiment(Model)


if __name__ == "__main__":
    app()
