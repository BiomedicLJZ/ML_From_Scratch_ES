# Regresión lineal

La regresión lineal predice un número continuo a partir de una o más características de
entrada, ajustando una línea recta (o un hiperplano) a los datos. Es el algoritmo supervisado
de **regresión** más simple, y todos los demás modelos de este curso reutilizan ideas
introducidas aquí: una función de hipótesis, una función de costo y el descenso de gradiente.

## El modelo

Dada una matriz de características `X` (de forma `n_muestras x n_características`), un vector
de pesos `w` y un sesgo escalar `b`, el modelo predice:

```
y_pred = X . w + b
```

Esto es simplemente una suma ponderada de las características más un desplazamiento.

## La función de costo

Para medir cuán equivocadas están las predicciones, usamos el **Error Cuadrático Medio (MSE)**:

```
MSE(w, b) = (1 / n) * suma((y_pred_i - y_i) ** 2)
```

Elevar al cuadrado los errores penaliza más los errores grandes que los pequeños, y mantiene la
función suave y diferenciable, algo importante para el siguiente paso.

## Descenso de gradiente

Queremos encontrar los valores de `w` y `b` que minimizan el MSE. Existe una solución de forma
cerrada (la ecuación normal), pero aquí usaremos el **descenso de gradiente**: un algoritmo
iterativo que empuja repetidamente a `w` y `b` en la dirección que reduce el costo más
rápidamente.

Los gradientes del MSE respecto a los parámetros son:

```
dw = (2 / n) * X.T . (y_pred - y)
db = (2 / n) * suma(y_pred - y)
```

Cada iteración actualiza los parámetros usando una **tasa de aprendizaje** `lr`:

```
w = w - lr * dw
b = b - lr * db
```

Repite esto durante un número fijo de iteraciones (o hasta que el costo deje de mejorar), y
`w`, `b` convergen hacia los valores que minimizan el MSE.

## Lo que vas a construir

A lo largo de las siguientes tareas implementarás, pieza por pieza, una clase
`LinearRegression` con:

1. `predict(X)` — la función de hipótesis.
2. `compute_cost(X, y)` — el MSE.
3. `compute_gradients(X, y)` — los gradientes del MSE.
4. `fit(X, y)` — el bucle de entrenamiento que une todo con descenso de gradiente.

Al final, tendrás un modelo de regresión lineal funcional construido enteramente con NumPy, sin
usar scikit-learn.
