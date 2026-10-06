import numpy as np


class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def _gini(self, y):
        counts = np.bincount(y)
        proportions = counts / len(y)
        return 1 - np.sum(proportions ** 2)

    def _split_dataset(self, X, y, feature_index, threshold):
        mask = X[:, feature_index] <= threshold
        return X[mask], y[mask], X[~mask], y[~mask]


if __name__ == '__main__':
    tree = DecisionTreeClassifier()
    X = np.array([[1.0], [2.0], [8.0], [9.0]])
    y = np.array([0, 0, 1, 1])
    print(tree._split_dataset(X, y, 0, 5.0))
