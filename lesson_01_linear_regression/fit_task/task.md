# Implementar `fit`

Une todas las piezas. `fit` entrena el modelo sobre `(X, y)` usando descenso de gradiente:

1. Inicializa `self.weights` como un vector de ceros de forma `(n_features,)` y `self.bias`
   en `0.0`.
2. Repite `self.n_iterations` veces:
   - Calcula `dw, db = self.compute_gradients(X, y)`.
   - Actualiza `self.weights -= self.learning_rate * dw`.
   - Actualiza `self.bias -= self.learning_rate * db`.
3. Devuelve `self` (esto permite encadenar `model.fit(X, y).predict(X_test)`).

Una vez que esto funcione, tendrás un modelo de regresión lineal completo entrenado desde cero
con NumPy. Pruébalo con los datos de `main.py`: `y = 2x + 1`, así que después de entrenar,
`weights` debería acercarse a `[2.0]` y `bias` a `1.0`.
