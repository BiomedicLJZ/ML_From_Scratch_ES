# Implementar `fit`

Une los dos pasos. `fit` recibe los datos `X` y `initial_centroids` de forma explícita (sin
inicialización aleatoria, así los resultados son reproducibles), y alterna asignación/
actualización durante `self.n_iterations` rondas:

1. `centroids = np.array(initial_centroids, dtype=float)` — copia para no mutar el array de
   quien llama.
2. Repite `self.n_iterations` veces:
   - `labels = self._assign_clusters(X, centroids)`.
   - `centroids = self._update_centroids(X, labels, len(centroids))`.
3. Guarda `self.centroids_ = centroids` y `self.labels_ = labels`.
4. Devuelve `self`.

Nota: como los clusters dejan de cambiar una vez que convergen, ejecutar iteraciones extra
después de la convergencia es inofensivo — la asignación y los centroides simplemente dejan de
moverse.
