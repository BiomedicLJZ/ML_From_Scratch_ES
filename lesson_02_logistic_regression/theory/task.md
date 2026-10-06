# Regresión logística

La regresión logística es la contraparte supervisada de **clasificación** de la regresión
lineal. Predice la probabilidad de que una muestra pertenezca a la clase `1` (frente a la clase
`0`), aplastando una combinación lineal de las características mediante una curva que siempre
produce un valor entre 0 y 1.

## La función sigmoide

```
sigmoid(z) = 1 / (1 + exp(-z))
```

`sigmoid` mapea cualquier número real al intervalo `(0, 1)`: entradas muy negativas van a ~0,
entradas muy positivas van a ~1, y `sigmoid(0) = 0.5`.

## El modelo

```
z = X . weights + bias
p = sigmoid(z)
```

`p` se interpreta como la probabilidad predicha de la clase `1`. Para obtener una etiqueta de
clase concreta, la comparamos con un umbral (comúnmente 0.5):

```
y_pred = 1 si p >= threshold, si no 0
```

## La función de costo

Aquí no podemos usar el Error Cuadrático Medio — produce una superficie de costo no convexa con
este modelo. En su lugar usamos la **entropía cruzada binaria**:

```
cost = -(1 / n) * suma(y * log(p) + (1 - y) * log(1 - p))
```

Esto penaliza fuertemente las predicciones confiadas y equivocadas. Un detalle práctico: si `p`
llega a ser exactamente 0 o 1, `log(p)` o `log(1 - p)` se vuelve `-infinito`. Para evitarlo,
**recortamos** (clip) `p` a un pequeño margen antes de 0 y de 1 antes de tomar el logaritmo.

## Descenso de gradiente

Sorprendentemente, el gradiente de la entropía cruzada binaria tiene exactamente la misma forma
que el gradiente del MSE de la regresión lineal, solo que con `p = sigmoid(X . weights + bias)`
en lugar de la predicción lineal directa:

```
dw = (1 / n) * X.T . (p - y)
db = (1 / n) * suma(p - y)
```

Entrenar significa de nuevo dar pasos repetidos en `weights` y `bias` en la dirección que reduce
el costo.

## Lo que vas a construir

Una clase `LogisticRegression` con `_sigmoid`, `predict_proba`, `predict`, `compute_cost`,
`compute_gradients` y `fit` — una pieza por tarea, exactamente como en regresión lineal.
