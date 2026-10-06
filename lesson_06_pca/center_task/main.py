import numpy as np


class PCA:
    def __init__(self, n_components=1):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def _center(self, X):
        return X - self.mean_


if __name__ == '__main__':
    pca = PCA()
    pca.mean_ = np.array([1.0, 2.0])
    X = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.0]])
    print(pca._center(X))
