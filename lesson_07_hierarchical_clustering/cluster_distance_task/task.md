# Implementar `_cluster_distance`

Implementa `_cluster_distance(cluster_a, cluster_b, distances)`, la distancia **single-linkage**
entre dos clusters: la distancia más pequeña entre cualquier punto de `cluster_a` y cualquier
punto de `cluster_b`.

`cluster_a` y `cluster_b` son listas de índices de puntos, y `distances` es la matriz completa
de distancias por pares de `_pairwise_distances` (ya implementada). `distances[np.ix_(cluster_a,
cluster_b)]` selecciona la submatriz de distancias entre cada punto de `cluster_a` y cada punto
de `cluster_b`; toma su `.min()`.
