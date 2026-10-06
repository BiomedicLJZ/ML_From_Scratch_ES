import numpy as np


class PCA:
    def __init__(self, n_components=1):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def _center(self, X):
        return X - self.mean_

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_centered = self._center(X)
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        return self

    def transform(self, X):
        return self._center(X).dot(self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    t = np.linspace(-5, 5, 50)
    noise = rng.normal(0, 0.01, size=50)
    X = np.column_stack([t, 0.3 * t + noise])
    pca = PCA(n_components=1)
    reduced = pca.fit_transform(X)
    print(reduced.shape)
    print(pca.transform(pca.mean_.reshape(1, -1)))
