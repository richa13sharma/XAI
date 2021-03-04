from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
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


def svd(data, components: int = 2):
    """
    Function that takes x data and returns transformed x data with specified components using SVD
    `data` is the x data to be transformed
    `components` are the number of components in the output df
    Returns:
    `df` a dataframe, transformed using PCA with the given number of components
    """
    svd = TruncatedSVD(n_components=components)
    svd.fit(data)
    transformed = svd.transform(data)
    df = pd.DataFrame(data=transformed)

    return df


def auto_pca(data):
    pca = PCA(n_components="mle", svd_solver="full")
    pca.fit(data)
    transformed = pca.transform(data)
    df = pd.DataFrame(data=transformed)

    eigenVectors = pca.components_

    return df, eigenVectors
