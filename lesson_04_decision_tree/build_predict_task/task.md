# Implementar `_build_tree` y `_predict_one`

Ahora une todo en un árbol entrenado que funcione.

## `_build_tree(X, y, depth)`

Construye el árbol de forma recursiva. Un nodo se almacena como un `dict` simple:
- Una **hoja**: `{'leaf': True, 'class': <clase mayoritaria>}`.
- Un **nodo interno**: `{'leaf': False, 'feature': ..., 'threshold': ..., 'left': ..., 'right': ...}`.

Pasos:
1. Condiciones de parada — devuelve una hoja con la clase mayoritaria
   (`np.bincount(y).argmax()`) si:
   - todas las muestras ya tienen la misma clase (`len(np.unique(y)) == 1`), o
   - `depth >= self.max_depth`, o
   - `len(y) < self.min_samples_split`.
2. En caso contrario, calcula `feature, threshold = self._best_split(X, y)`. Si `feature is
   None` (ninguna división ayuda), devuelve también una hoja con la clase mayoritaria.
3. En caso contrario, divide los datos con `_split_dataset`, y devuelve un nodo interno cuyos
   `'left'` y `'right'` son los resultados de llamar a `_build_tree` recursivamente sobre cada
   mitad, en `depth + 1`.

## `_predict_one(x, node)`

Recorre el árbol para una sola muestra `x`, empezando en `node`:
1. Si `node['leaf']` es `True`, devuelve `node['class']`.
2. En caso contrario, compara `x[node['feature']]` con `node['threshold']`: si `<=`, recurre en
   `node['left']`; si no, recurre en `node['right']`.

`fit` y `predict` ya están implementados para ti — `fit` llama a `_build_tree` y guarda el
resultado en `self.tree_`, y `predict` llama a `_predict_one` una vez por fila.
