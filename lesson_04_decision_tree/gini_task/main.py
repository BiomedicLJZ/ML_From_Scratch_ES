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


if __name__ == '__main__':
    tree = DecisionTreeClassifier()
    print(tree._gini(np.array([0, 0, 1, 1])))
    print(tree._gini(np.array([0, 0, 0, 0])))
