# DBSCAN

Todos los algoritmos de clustering vistos hasta ahora en este curso (K-Means, clustering
jerárquico) necesitan que se les diga cuántos clusters buscar, y asumen que los clusters son
más o menos manchas redondas. **DBSCAN** (Density-Based Spatial Clustering of Applications with
Noise) no hace ninguna de las dos cosas: encuentra clusters de forma arbitraria observando qué
tan densamente empaquetados están los puntos, y puede etiquetar como **ruido** a los puntos que
no pertenecen a ningún lado, en lugar de forzarlos dentro de un cluster.

## Dos parámetros

- `eps` — el radio de un vecindario alrededor de un punto.
- `min_samples` — cuántos puntos (incluyendo al propio punto) deben estar dentro de ese radio
  para que el punto cuente como "denso".

## Tres tipos de puntos

- **Punto núcleo (core point)** — tiene al menos `min_samples` puntos (incluyéndose a sí mismo)
  dentro de la distancia `eps`.
- **Punto frontera (border point)** — no es un punto núcleo en sí mismo, pero cae dentro de
  `eps` de un punto núcleo.
- **Punto de ruido (noise point)** — ni es un punto núcleo, ni está dentro de `eps` de ningún
  punto núcleo.

## Cómo se forman los clusters

DBSCAN visita cada punto (en orden de índice, para que el proceso sea repetible). Si un punto ya
forma parte de un cluster, lo omite. En caso contrario, examina el vecindario del punto:

- Si el vecindario es demasiado pequeño (menos de `min_samples` puntos), el punto se marca como
  ruido **por ahora** — todavía podría ser reclamado más tarde como punto frontera de algún otro
  cluster encontrado cerca.
- Si el punto es un punto núcleo, DBSCAN inicia un nuevo cluster y lo **expande**: cada punto del
  vecindario se une al cluster, y si alguno de esos puntos es *también* un punto núcleo, su
  vecindario también se incorpora. Esta reacción en cadena se llama **densidad-alcanzabilidad**:
  un cluster es todo lo que se puede alcanzar desde un punto núcleo saltando a través de los
  vecindarios de otros puntos núcleo.

Los puntos a los que ninguna expansión llega nunca conservan su etiqueta de ruido, `-1`.

## Lo que vas a construir

1. `_region_query(X, point_idx)` — encuentra cada punto dentro de `eps` de un punto dado.
2. `_expand_cluster(...)` — hace crecer un cluster hacia afuera desde un punto núcleo usando la
   reacción en cadena descrita arriba.
3. `fit(X)` — el bucle principal: visita los puntos en orden, omite los ya visitados, marca los
   vecindarios pequeños como ruido, y llama a `_expand_cluster` cada vez que un punto núcleo
   inicia un nuevo cluster.

Al final tendrás un algoritmo de clustering basado en densidad que funciona, sin necesitar
ningún parámetro `n_clusters` — descubre los clusters (y los valores atípicos) a partir de la
densidad de los propios datos.
