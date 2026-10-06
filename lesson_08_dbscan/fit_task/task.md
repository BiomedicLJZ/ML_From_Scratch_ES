# Implementar `fit`

Une `_region_query` y `_expand_cluster` en el algoritmo DBSCAN completo.

1. `labels = np.full(n, -1, dtype=int)` — todos empiezan como ruido.
2. `visited = np.zeros(n, dtype=bool)` — para rastrear qué puntos ya se examinaron.
3. `cluster_id = 0`.
4. Para `point_idx` desde `0` hasta `n - 1` (en orden, para que el resultado sea repetible):
   - Omítelo si ya fue `visited`.
   - Márcalo como `visited`.
   - Calcula su vecindario con `_region_query`.
   - Si el vecindario es más pequeño que `self.min_samples`, no es un punto núcleo — déjalo como
     ruido por ahora (todavía podría ser reclamado más tarde como punto frontera de otro cluster)
     y continúa.
   - En caso contrario, SÍ es un punto núcleo: llama a `_expand_cluster(X, labels, point_idx,
     neighbors, cluster_id)` para hacer crecer un cluster completamente nuevo desde él, marca
     como `visited` cada punto de su vecindario, e incrementa `cluster_id` para el siguiente
     cluster.
5. Guarda el array final en `self.labels_` y devuelve `self`.

Pruébalo con el ejemplo de `main.py`: dos grupos densos de 4 puntos muy separados entre sí, más
un punto aislado. Deberías obtener un id de cluster para cada grupo, y `-1` para el punto
aislado.
