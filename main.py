import typer
import modules
import perturb
import utils

app = typer.Typer()


@app.command()
def main(
    create: bool = typer.Option(False, help="Create, train and save model"),
    score: bool = typer.Option(False, help="Compute model score"),
    with_pert: bool = typer.Option(False, help="Compute model score"),
    model_path: str = typer.Argument("./obj/v1.h5", help="Path to saved model"),
    num_PCs: int = typer.Option(False, help="Number of principal components"),
    csv_path: str = typer.Option("./data/data.csv", help="Path to dataset csv"),
    auto_pca: bool = typer.Option(False, help="Perform PCA - determine dims using mle"),
):

    dataset = modules.DataSet(
        path=csv_path,
        train_split_percentage=0.8,
        num_PCs=num_PCs,
        auto_pca=auto_pca,
    )
    X_train, y_train, X_test, y_test = dataset.load()

    Model = modules.KerasModel((X_test, y_test), (X_train, y_train))

    # Note: If you're using PCA(auto or not), --create is needed when the dataset is changed.
    if create:
        Model.create(dataset.dims)
        Model.train(epochs=30)
        Model.save(model_path)

    Model.load(model_path)

    if score:
        print("Score: ", Model.test())

    if with_pert:
        perturb.perturbateExperiment(Model)


if __name__ == "__main__":
    app()
