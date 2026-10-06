# Implementar `_assign_clusters`

Implementa el paso de asignación. Para cada punto en `X`, encuentra el índice del centroide más
cercano en `centroids` (distancia euclidiana) y devuelve el array de esos índices, uno por punto
de `X`.

`X` tiene forma `(n_samples, n_features)`, `centroids` tiene forma `(n_clusters, n_features)`.
Una forma vectorizada de hacerlo sin un bucle explícito:

```
distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
```

`distances` tiene forma `(n_samples, n_clusters)`: `distances[i, k]` es la distancia del punto
`i` al centroide `k`. Devuelve `np.argmin(distances, axis=1)`, el índice del centroide más
cercano para cada fila.
