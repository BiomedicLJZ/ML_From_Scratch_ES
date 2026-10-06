# Clustering jerárquico (aglomerativo)

K-Means requiere que decidas el número de clusters de antemano y parte de centroides
aleatorios. El **clustering aglomerativo** toma un enfoque distinto: empieza con cada punto
como su propio cluster, y luego fusiona repetidamente los dos clusters más cercanos, una fusión
a la vez, hasta que queda el número deseado de clusters. La secuencia de fusiones puede
dibujarse como un árbol (un **dendrograma**), donde puedes "cortar" a cualquier altura para
obtener cualquier número de clusters — pero en esta lección solo construiremos el algoritmo de
fusión en sí, no el dibujo.

## Linkage: ¿cómo se mide la distancia entre dos *clusters*?

Una vez que un cluster tiene más de un punto, "la distancia entre dos clusters" es ambigua —
¿te refieres al par de puntos más cercano, al más lejano, o al promedio? Las distintas opciones
se llaman diferentes criterios de **linkage**:

- **Single linkage**: la distancia entre dos clusters es la distancia *mínima* entre cualquier
  punto de uno y cualquier punto del otro.
- **Complete linkage**: la distancia *máxima* entre cualquier par.
- **Average linkage**: la distancia *promedio* sobre todos los pares.

En esta lección implementaremos únicamente **single linkage**. Tiende a producir clusters
largos y encadenados, y es el más simple de razonar: dos clusters están "cerca" tan pronto como
cualquiera de sus puntos lo está.

## El algoritmo

```
clusters = un cluster por punto
mientras numero_de_clusters > n_clusters:
    encuentra los dos clusters con la menor distancia single-linkage entre ellos
    fusionalos en un solo cluster
asigna a cada punto el id del cluster final en el que terminó
```

## Desempate

Los conjuntos de datos reales pueden tener empates (dos pares distintos de clusters a
exactamente la misma distancia). Para mantener el algoritmo determinista, este curso usa una
regla fija: recorre los pares de clusters `(a, b)` con `a < b` en orden, y solo reemplaza el
mejor par actual cuando se encuentra una distancia estrictamente menor. Esto significa que,
entre pares empatados, siempre gana el primero encontrado en ese orden de recorrido.

## Lo que vas a construir

1. `_pairwise_distances(X)` — la matriz completa de distancias entre cada par de puntos.
2. `_cluster_distance(cluster_a, cluster_b, distances)` — la distancia single-linkage entre dos
   clusters.
3. `fit(X)` — el bucle de fusión que une todo y produce `self.labels_`.
