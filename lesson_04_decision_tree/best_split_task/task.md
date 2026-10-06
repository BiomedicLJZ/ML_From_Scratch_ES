# Implementar `_best_split`

Implementa `_best_split(X, y)`, que busca la mejor pregunta posible para hacer en un nodo: el
par `(feature_index, threshold)` cuya división da la menor **impureza de Gini ponderada**:

```
weighted_gini = (n_izquierda / n) * Gini(y_izquierda) + (n_derecha / n) * Gini(y_derecha)
```

Pasos:
1. Inicializa `best_feature = None`, `best_threshold = None`, `best_gini = inf`.
2. Recorre cada índice de característica `0 .. n_features - 1`, en orden.
3. Para cada característica, recorre cada **valor único** de esa característica
   (`np.unique(X[:, feature_index])`), en orden ascendente, como umbral candidato.
4. Para cada candidato, divide los datos con `_split_dataset` (ya implementado). Si algún lado
   queda vacío, sáltalo — no es una división real.
5. Calcula el Gini ponderado de la división usando `_gini` (ya implementado). Si es
   **estrictamente menor** que `best_gini`, actualiza `best_feature`, `best_threshold`,
   `best_gini`.
6. Devuelve `best_feature, best_threshold`.

Usar una comparación estricta `<` y recorrer características/umbrales en un orden fijo significa
que, si varias divisiones empatan en el mejor puntaje, siempre obtienes la misma respuesta
reproducible: la primera encontrada.

Si no existe ninguna división válida (p. ej. todas las filas son idénticas), esto devuelve
`(None, None)`.
