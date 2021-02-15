from sklearn.decomposition import PCA
import pandas as pd


def pca(data, components: int = 2):
    """
    Function that takes x data and returns transformed x data with specified components using PCA
    `data` is the x data to be transformed
    `components` are the number of components in the output df
    Returns:
    `df` a dataframe, transformed using PCA with the given number of components
    """
    pca = PCA(n_components=components)
    components = pca.fit_transform(data)
    df = pd.DataFrame(data=components)

    return df
