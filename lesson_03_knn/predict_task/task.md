# Implementar `predict`

Implementa `predict`, que clasifica un lote completo de puntos a la vez. Aplica
`self._predict_one` a cada fila de `X` y reúne los resultados en un array de NumPy.

Este es el método que realmente usa quien utiliza tu clase — `_euclidean_distance` y
`_predict_one` son ayudantes internos (de ahí el guion bajo inicial).
