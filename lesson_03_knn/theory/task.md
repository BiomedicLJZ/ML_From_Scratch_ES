# K-Nearest Neighbors

K-Nearest Neighbors (KNN) es un algoritmo supervisado de **clasificación** que no asume nada
sobre la forma de la frontera de decisión. En lugar de aprender un conjunto de parámetros como
la regresión lineal o logística, simplemente memoriza los datos de entrenamiento y, al momento
de predecir, observa los `k` puntos de entrenamiento más cercanos para decidir la etiqueta de
un punto nuevo.

## "Entrenar" es solo memorizar

El paso `fit` de KNN no calcula nada. Simplemente almacena las características y etiquetas de
entrenamiento para usarlas después. Todo el trabajo real ocurre durante la predicción. Esto
convierte a KNN en un **aprendiz perezoso** (lazy learner): no hay costo de entrenamiento por
adelantado, pero cada predicción tiene que recorrer todo el conjunto de entrenamiento.

## Midiendo la "cercanía"

Necesitamos una métrica de distancia entre dos vectores de características `a` y `b`. La opción
más común es la **distancia euclidiana**:

```
distance(a, b) = sqrt(sum((a - b) ** 2))
```

## Haciendo una predicción

Para clasificar un punto nuevo `x`:

1. Calcula la distancia de `x` a cada punto del conjunto de entrenamiento.
2. Encuentra los `k` puntos de entrenamiento con la menor distancia (los "vecinos más
   cercanos").
3. Consulta sus etiquetas y devuelve la más común (voto mayoritario).

## Lo que vas a construir

Implementarás un `KNNClassifier` con:

1. `_euclidean_distance(a, b)` — la métrica de distancia.
2. `_predict_one(x)` — la lógica completa de predicción para un solo punto.
3. `predict(X)` — aplica `_predict_one` a cada fila de un lote.

## Más allá de la clasificación

La misma idea funciona para **regresión**: en lugar de un voto mayoritario sobre las etiquetas
de los vecinos, promediarías los valores objetivo de los vecinos. Aquí solo implementamos la
versión de clasificación, pero ten en cuenta esta extensión — encaja de forma natural con el
hábito de "generalizar la regla" que necesitarás a lo largo de este curso.
