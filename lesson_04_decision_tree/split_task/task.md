# Implementar `_split_dataset`

Implementa `_split_dataset(X, y, feature_index, threshold)`. Particiona el conjunto de datos en
dos grupos según una característica y un umbral:

- Grupo **izquierdo**: filas donde `X[:, feature_index] <= threshold`.
- Grupo **derecho**: todas las demás filas.

Pasos:
1. Construye una máscara booleana: `mask = X[:, feature_index] <= threshold`.
2. Devuelve `X[mask], y[mask], X[~mask], y[~mask]` — X izquierda, y izquierda, X derecha, y
   derecha, en ese orden.
