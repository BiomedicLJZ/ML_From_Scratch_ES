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

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_feature, best_threshold, best_gini = None, None, np.inf
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split_dataset(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                weighted_gini = (len(y_left) / n_samples) * self._gini(y_left) + \
                    (len(y_right) / n_samples) * self._gini(y_right)
                if weighted_gini < best_gini:
                    best_feature, best_threshold, best_gini = feature_index, threshold, weighted_gini
        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1 or depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'leaf': True, 'class': np.bincount(y).argmax()}
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {'leaf': True, 'class': np.bincount(y).argmax()}
        X_left, y_left, X_right, y_right = self._split_dataset(X, y, feature, threshold)
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X_left, y_left, depth + 1),
            'right': self._build_tree(X_right, y_right, depth + 1),
        }

    def fit(self, X, y):
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])


if __name__ == '__main__':
    X = np.array([[1.0, 1.0], [2.0, 1.0], [8.0, 1.0], [9.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    print(tree.predict(X))  # Deberia imprimir [0 0 1 1]
