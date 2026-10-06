# Implementar `compute_gradients`

La entropía cruzada binaria tiene un gradiente exactamente de la misma forma que el del MSE,
solo que con `predict_proba` en lugar de la predicción lineal directa:

```
dw = (1 / n) * X.T . (predict_proba(X) - y)
db = (1 / n) * suma(predict_proba(X) - y)
```

Pasos:
1. `n_samples = X.shape[0]`.
2. `errors = self.predict_proba(X) - y`.
3. `dw = (1 / n_samples) * X.T.dot(errors)`.
4. `db = (1 / n_samples) * np.sum(errors)`.
5. Devuelve `dw, db`.
