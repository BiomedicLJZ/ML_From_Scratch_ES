# Análisis de Componentes Principales (PCA)

PCA es el algoritmo clásico de **reducción de dimensionalidad no supervisada**. En lugar de
predecir una etiqueta, redescribe los datos usando menos números mientras conserva la mayor
cantidad posible de información (varianza).

## La idea

Los conjuntos de datos reales suelen tener características que varían juntas. PCA encuentra
nuevos ejes, llamados **componentes principales**, que apuntan en las direcciones donde los
datos varían más. El primer componente principal es la dirección de máxima varianza; el segundo
es la dirección de la siguiente mayor varianza que es perpendicular al primero, y así
sucesivamente.

Proyectar los datos sobre solo los primeros componentes principales da una versión de menor
dimensión del conjunto de datos que aún captura la mayor parte de la dispersión original.

## Las matemáticas

1. **Centrar los datos.** Resta la media de cada característica para que los datos queden
   centrados en el origen:
   ```
   X_centered = X - mean(X)
   ```
2. **Calcular la matriz de covarianza** de los datos centrados. Sus entradas describen cuánto
   varía junto cada par de características.
3. **Encontrar los autovectores y autovalores** de la matriz de covarianza. Cada autovector es
   una dirección en el espacio de características; su autovalor es la cantidad de varianza a lo
   largo de esa dirección.
4. **Ordenar por autovalor, descendente**, y quedarse con los primeros `n_components`
   autovectores. Estos son los componentes principales.
5. **Proyectar** los datos centrados sobre los componentes:
   ```
   X_reduced = X_centered . components.T
   ```

## Una nota sobre el signo

Un autovector y su negación apuntan exactamente a la misma línea — representan el mismo eje. No
hay nada en las matemáticas que determine si un componente principal "apunta" hacia la izquierda
o la derecha a lo largo de ese eje, así que distintas implementaciones (¡correctas!) pueden
devolver componentes con signos opuestos. Esto es esperado y no es un error.

## Lo que vas a construir

Una clase `PCA` con:
1. `_center(X)` — resta la media.
2. `fit(X)` — calcula la media, la matriz de covarianza, la descomposición en autovalores/
   autovectores, y guarda los primeros `n_components` componentes y sus autovalores
   (`explained_variance_`).
3. `transform(X)` — proyecta nuevos datos sobre los componentes guardados.
