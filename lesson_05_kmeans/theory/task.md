# Clustering K-Means

K-Means es el primer algoritmo **no supervisado** de este curso: no hay un objetivo `y` que
predecir, solo una matriz de características `X`. El objetivo es agrupar las muestras en `k`
clusters de forma que los puntos del mismo cluster estén cerca entre sí.

## Centroides

Cada cluster está representado por un **centroide**: el punto en su centro. K-Means alterna
entre dos pasos hasta que los centroides dejan de moverse (o se alcanza un número fijo de
iteraciones):

1. **Paso de asignación** — asigna cada punto a su centroide más cercano (por distancia
   euclidiana).
2. **Paso de actualización** — recalcula cada centroide como la media de los puntos que tiene
   actualmente asignados.

## El objetivo

K-Means minimiza la **inercia** (también llamada suma de cuadrados dentro del cluster):

```
inercia = suma sobre todos los puntos de distancia_al_cuadrado(punto, su centroide asignado)
```

Cada uno de los dos pasos anteriores solo puede disminuir o mantener igual esta cantidad, por lo
que el algoritmo converge — aunque no necesariamente al óptimo global, ya que el resultado
depende de dónde empezaron los centroides.

## La inicialización importa

Como los clusters finales dependen de los centroides iniciales, en este curso `fit` recibe los
centroides iniciales como un argumento explícito en lugar de elegirlos al azar. En la práctica
(por ejemplo en scikit-learn), K-Means normalmente se ejecuta varias veces desde distintas
inicializaciones aleatorias y se conserva el mejor resultado (menor inercia) — pero eso es una
optimización sobre el algoritmo central que estás a punto de construir.

## Lo que vas a construir

1. `_assign_clusters(X, centroids)` — el paso de asignación.
2. `_update_centroids(X, labels, n_clusters)` — el paso de actualización.
3. `fit(X, initial_centroids)` — el bucle que une ambos pasos.
