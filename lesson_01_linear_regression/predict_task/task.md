# Implementar `predict`

Implementa el método `predict` de `LinearRegression`. Debe calcular la hipótesis del modelo:

```
y_pred = X . weights + bias
```

`X` es un array de NumPy de forma `(n_muestras, n_características)`, `self.weights` es un
array 1-D de forma `(n_características,)`, y `self.bias` es un escalar. Usa
`X.dot(self.weights)` para el producto matriz-vector, y luego suma el bias.

Devuelve el array 1-D resultante de predicciones, una por muestra.
