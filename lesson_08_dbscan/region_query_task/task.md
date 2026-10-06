# Implementar `_region_query`

Dado un punto (por su índice en `X`), encuentra cada punto de `X` a una distancia `eps` de él —
este es el vecindario que DBSCAN usa para decidir si un punto es un punto núcleo.

Pasos:
1. Calcula la distancia euclidiana de `X[point_idx]` a cada fila de `X` de una sola vez:
   `distances = np.sqrt(((X - X[point_idx]) ** 2).sum(axis=1))`.
2. Devuelve los índices donde esa distancia es a lo sumo `self.eps`, usando
   `np.where(distances <= self.eps)[0]`.

Ten en cuenta que `point_idx` en sí mismo siempre está incluido en el resultado (su distancia a
sí mismo es 0).
