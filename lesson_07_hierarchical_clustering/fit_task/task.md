# Implementar `fit`

Une todo en el algoritmo completo de clustering aglomerativo:

1. `n = X.shape[0]`, y `distances = self._pairwise_distances(X)`.
2. Empieza con un cluster por punto: `clusters = [[i] for i in range(n)]` (una lista de listas
   de índices de puntos).
3. Mientras `len(clusters) > self.n_clusters`:
   - Encuentra el par de clusters `(a, b)` (índices dentro de la lista `clusters`, con `a < b`)
     cuya `_cluster_distance` sea la más pequeña. Recorre `a` desde `0` hacia arriba y `b` desde
     `a + 1` hacia arriba, y actualiza tu "mejor hasta ahora" solo cuando encuentres una
     distancia *estrictamente menor* — esta es la regla de desempate de la página de teoría, y
     es lo que hace que el resultado sea determinista.
   - Fusiónalos: `clusters[a] = clusters[a] + clusters[b]`, luego `del clusters[b]`.
4. Una vez que termina el bucle, construye el array final `(n,)` de etiquetas: para la posición
   de cada cluster en la lista final `clusters` (su índice ahí es el id del cluster), asigna ese
   id a cada índice de punto que contiene.
5. Guarda el resultado en `self.labels_` y devuelve `self`.

Los puntos que terminan en el mismo cluster final compartirán la misma etiqueta — pero qué
número entero le toca a cada cluster es solo contabilidad, no algo con significado
independiente.
