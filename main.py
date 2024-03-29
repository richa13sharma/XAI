import typer
import modules
import perturb
import utils
from nbdt import NBDT

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
    nbdt: bool = typer.Option(False, help="Perform NBDT experiment"),
):

    dataset = modules.DataSet(
        path=csv_path, train_split_percentage=0.8, num_PCs=num_PCs, auto_pca=auto_pca
    )
    X_train, y_train, X_test, y_test = dataset.load()

    Model = modules.KerasModel((X_test, y_test), (X_train, y_train))

    # Note: If you're using PCA(auto or not), --create is needed when the dataset is changed.
    if create:
        Model.create(dataset.dims)
        Model.train(epochs=30)
        Model.save(model_path)

    Model.load(model_path)

    weights, bias = Model.get_Weights_Bias()
    input_vars = Model.X_train.values
    output_vars = Model.y_train.values
    LRP_Helper = modules.LRPHelper(weights, bias, input_vars, output_vars)

    if score:
        print("Score: ", Model.test())

    if with_pert:
        perturb.perturbateExperiment(Model)

    if nbdt:
        # TODO: Move this to separate file once logic gets complex
        decision_tree = NBDT(model=Model, topK=2)

        accuracy = decision_tree.create_DT()
        print("DT Accuracy:", accuracy)

        decision_tree.dump_rules()


if __name__ == "__main__":
    app()
