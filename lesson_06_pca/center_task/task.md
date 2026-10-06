# Implementar `_center`

PCA necesita que los datos estén centrados en el origen antes de calcular la varianza.
Implementa `_center(self, X)` para que devuelva `X` con `self.mean_` restado de cada fila.

Esto se usa tanto durante el entrenamiento (una vez que `self.mean_` se ha calculado a partir de
los datos de entrenamiento) como al transformar datos nuevos (usando la misma media con la que
se entrenó el modelo).
