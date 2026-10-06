# Implementar `compute_gradients`

El descenso de gradiente necesita saber, para los valores actuales de `weights` y `bias`, en
qué dirección aumenta el costo, para poder dar un paso en la dirección contraria. Implementa
`compute_gradients`, que devuelve los gradientes del MSE respecto a `weights` (`dw`) y `bias`
(`db`):

```
dw = (2 / n) * X.T . (y_pred - y)
db = (2 / n) * suma(y_pred - y)
```

Pasos:
1. `n_samples = X.shape[0]`.
2. Calcula `predictions = self.predict(X)` y `errors = predictions - y`.
3. `dw = (2 / n_samples) * X.T.dot(errors)`.
4. `db = (2 / n_samples) * np.sum(errors)`.
5. Devuelve `dw, db` como una tupla.
