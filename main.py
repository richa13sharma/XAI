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

    op = LRP_Helper.create_DT_inputs(5, input_vars, output_vars)
    print(len(op), len(op[0]))  # 800 15(k=5 * numberof layers=3)

    if score:
        print("Score: ", Model.test())

    if with_pert:
        perturb.perturbateExperiment(Model)

    if nbdt:
        # TODO: Move this to separate file once logic gets complex
        Decision_tree = NBDT(model=Model)
        f1l2n3 = Decision_tree.get_step_relevance_for_neuron(
            feature=1, layer=2, neuron=3
        )
        # f1l2n3 = [0.022398318930674618, 0.022527274925784792, 0.022671048631382168,
        #       0.022841418427102942, 0.023037959696827447, 0.02319901032007013,
        #       0.023319496901534998, 0.023403225902563808, 0.023496894778267167, 0.02335571075813635]
        examplef1l2n3 = [1, 1, 7, 8, 4, 3, 1, 1, 1, 4]
        res = Decision_tree.generate_step_range_pairs(examplef1l2n3, 3)
        print(res)
        # res = ([(0.1, 0.2), (0.6, 0.9)], [(0.3, 0.5), (1.0, 1.0)])


if __name__ == "__main__":
    app()
