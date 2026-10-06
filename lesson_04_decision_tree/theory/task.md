# Árboles de decisión

Un árbol de decisión es un modelo supervisado de **clasificación** (o regresión) que predice
haciendo una secuencia de preguntas de sí/no sobre las características, una a la vez, hasta
llegar a una respuesta. Cada nodo interno pregunta "¿la característica `i` es <= algún umbral?",
y dirige la muestra a la izquierda o a la derecha según la respuesta. Cada hoja contiene una
clase predicha.

A diferencia de la regresión lineal/logística, un árbol de decisión no se entrena con descenso
de gradiente. Se construye de forma voraz (greedy) y recursiva: en cada nodo, se elige la mejor
pregunta posible en ese momento.

## Midiendo qué tan "mezclado" está un nodo: la impureza de Gini

Para decidir cuál pregunta es mejor, necesitamos una forma de medir qué tan mezcladas están las
clases dentro de un grupo de muestras. Usamos la **impureza de Gini**:

```
Gini(y) = 1 - suma(p_c ** 2)      para cada clase c, donde p_c es la fracción de muestras de la clase c
```

Un nodo que contiene una sola clase tiene `Gini = 0` (puro). Un nodo dividido 50/50 entre dos
clases tiene `Gini = 0.5`. Cuanto menor, mejor.

## Eligiendo la mejor división

Para cada característica y cada umbral candidato, podemos dividir las muestras en un grupo
"izquierdo" (`feature <= threshold`) y un grupo "derecho" (`feature > threshold`), y calcular la
impureza de Gini **ponderada** de los dos grupos:

```
weighted_gini = (n_izquierda / n) * Gini(y_izquierda) + (n_derecha / n) * Gini(y_derecha)
```

La mejor división es el par (característica, umbral) que da la menor impureza de Gini
ponderada. En este curso, buscamos entre todas las características, y entre todos los valores
únicos de esa característica como umbral candidato, y siempre resolvemos los empates tomando la
primera división encontrada (menor índice de característica, luego menor umbral), de modo que
el resultado sea completamente determinista.

## Construyendo el árbol

Partiendo de la raíz (todas las muestras de entrenamiento), de forma recursiva:
1. Si todas las muestras de este nodo tienen la misma clase, o el nodo es demasiado pequeño, o
   se alcanzó la profundidad máxima, detente y convierte este nodo en una **hoja** que contiene
   la clase mayoritaria.
2. En caso contrario, encuentra la mejor división, particiona las muestras en izquierda/derecha,
   y recurre en dos nodos hijos.

## Lo que vas a construir

Implementarás un `DecisionTreeClassifier` con:
1. `_gini(y)` — la medida de impureza.
2. `_split_dataset(X, y, feature_index, threshold)` — particiona los datos según una pregunta
   candidata.
3. `_best_split(X, y)` — busca la mejor pregunta para hacer en un nodo.
4. `_build_tree(X, y, depth)` y `_predict_one(x, node)` — construyen el árbol de forma
   recursiva, y lo recorren para hacer predicciones.

Una muestra va a la **izquierda** en un nodo si `x[feature] <= threshold`, y a la **derecha** en
caso contrario.
