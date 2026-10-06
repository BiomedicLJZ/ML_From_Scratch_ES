# Implementar `predict_proba`

Combina el modelo lineal con la sigmoide que acabas de escribir para obtener una probabilidad
predicha:

```
z = X . weights + bias
p = sigmoid(z)
```

Usa `self._sigmoid(...)` (ya implementada) aplicada a `X.dot(self.weights) + self.bias`.
Devuelve el array resultante de probabilidades, una por muestra.
