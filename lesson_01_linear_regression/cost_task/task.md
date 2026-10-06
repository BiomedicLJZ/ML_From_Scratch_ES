# Implementar `compute_cost`

Implementa `compute_cost`, que mide qué tan lejos están las predicciones del modelo de los
valores reales usando el **Error Cuadrático Medio**:

```
MSE = (1 / n) * suma((y_pred - y) ** 2)
```

Pasos:
1. Obtén las predicciones del modelo para `X` llamando a `self.predict(X)` (ya implementado).
2. Calcula el error elemento a elemento: `predictions - y`.
3. Devuelve la media de los errores al cuadrado con `np.mean(errors ** 2)`.
