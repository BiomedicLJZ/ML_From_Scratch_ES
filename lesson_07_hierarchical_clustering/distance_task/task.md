# Implementar `_pairwise_distances`

Antes de poder fusionar clusters, necesitamos conocer la distancia euclidiana entre cada par de
puntos. Implementa `_pairwise_distances(X)`, que devuelve una matriz `(n_samples, n_samples)`
donde la entrada `[i, j]` es la distancia euclidiana entre el punto `i` y el punto `j` (la
diagonal es toda ceros, y la matriz es simétrica).

Una forma completamente vectorizada de hacerlo con broadcasting de NumPy:

```
X[:, None, :] - X[None, :, :]
```

produce un array `(n_samples, n_samples, n_features)` con todas las diferencias por pares.
Elévalo al cuadrado, suma sobre el último eje, y toma la raíz cuadrada para obtener la matriz de
distancias.
