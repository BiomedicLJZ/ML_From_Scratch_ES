# Implementar `_predict_one`

Implementa la lógica completa de predicción para un solo punto `x`:

1. Calcula la distancia de `x` a cada fila de `self.X_train`, usando `self._euclidean_distance`
   (ya implementado).
2. Encuentra los índices de las `self.n_neighbors` distancias más pequeñas con
   `np.argsort(distances)[:self.n_neighbors]`.
3. Consulta las etiquetas correspondientes en `self.y_train`.
4. Devuelve la etiqueta más común entre ellas, usando
   `Counter(nearest_labels.tolist()).most_common(1)[0][0]` (`Counter` ya está importado al
   inicio de `main.py`).
