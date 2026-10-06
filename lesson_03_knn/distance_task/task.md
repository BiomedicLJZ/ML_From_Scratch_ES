# Implementar `_euclidean_distance`

Implementa la distancia euclidiana entre dos vectores de características 1-D `a` y `b`:

```
distance(a, b) = sqrt(sum((a - b) ** 2))
```

Usa `np.sqrt` y `np.sum` sobre la diferencia al cuadrado elemento a elemento `(a - b) ** 2`.

Nota que `fit`, arriba, no hace más que almacenar los datos de entrenamiento — KNN no tiene una
fase de entrenamiento real, todo el trabajo ocurre cuando predices.
