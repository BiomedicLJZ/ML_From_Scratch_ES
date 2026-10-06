# Implementar `fit`

Exactamente igual que el `fit` de la regresión lineal, pero ahora los gradientes vienen de
`compute_gradients` para regresión logística:

1. Inicializa `self.weights` en ceros de forma `(n_features,)` y `self.bias` en `0.0`.
2. Repite `self.n_iterations` veces:
   - `dw, db = self.compute_gradients(X, y)`.
   - `self.weights -= self.learning_rate * dw`.
   - `self.bias -= self.learning_rate * db`.
3. Devuelve `self`.

Pruébalo con los datos de la demo en `main.py`: dos puntos cerca de 0 (clase 0) y dos puntos
cerca de 9-10 (clase 1). Después de entrenar, `predict` debería clasificar los cuatro
correctamente.
