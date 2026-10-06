# Implementar `_expand_cluster`

`point_idx` es un punto núcleo que inicia un cluster completamente nuevo `cluster_id`, y
`neighbors` es su vecindario ya calculado (de `_region_query`). Haz crecer el cluster hacia
afuera a través de cada punto alcanzable mediante una cadena de puntos núcleo.

Usa un enfoque basado en cola (recorrido en anchura):

1. Etiqueta a `point_idx` mismo con `cluster_id`.
2. Pon `neighbors` en una cola, y mantén un `set` con todo lo que alguna vez se agregó a la
   cola (para nunca encolar el mismo punto dos veces).
3. Procesa la cola un punto a la vez (`current`):
   - Si `current` todavía no tiene etiqueta (`labels[current] == -1`), asígnale `cluster_id`.
   - Luego verifica si `current` es *también* un punto núcleo: calcula su propio vecindario con
     `_region_query`, y si tiene al menos `self.min_samples` vecinos, encola aquellos vecinos que
     aún no estén en el `set` de la cola.
4. Detente cuando la cola se agote.

De esta manera, el cluster crece a través de cada punto núcleo que encuentra, y los puntos
frontera (puntos que no son núcleo pero están simplemente cerca de un punto núcleo) se
incorporan y etiquetan, pero no disparan por sí mismos más crecimiento.
