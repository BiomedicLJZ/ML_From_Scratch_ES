# Implementar `_update_centroids`

Implementa el paso de actualización. Dada la asignación actual de clusters `labels` (un entero
en `0..n_clusters-1` por fila de `X`), recalcula cada centroide como la media de los puntos que
tiene actualmente asignados.

Para el cluster `k`, su nuevo centroide es `X[labels == k].mean(axis=0)`. Devuelve un array de
forma `(n_clusters, n_features)` apilando el nuevo centroide para cada `k` en
`range(n_clusters)`, por ejemplo con `np.array([... for k in range(n_clusters)])`.

Puedes asumir que cada cluster tiene al menos un punto asignado (los datos de prueba están
construidos así).
