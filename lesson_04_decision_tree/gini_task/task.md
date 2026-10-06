# Implementar `_gini`

Implementa `_gini(y)`, la impureza de Gini de un grupo de etiquetas:

```
Gini(y) = 1 - suma(p_c ** 2)
```

donde `p_c` es la fracción de muestras que pertenecen a la clase `c`.

Pasos:
1. Cuenta cuántas muestras pertenecen a cada clase con `np.bincount(y)` (esto asume que las
   etiquetas son enteros pequeños no negativos, p. ej. `0, 1, 2, ...`).
2. Divide entre `len(y)` para obtener las proporciones de clase.
3. Devuelve `1 - np.sum(proportions ** 2)`.

Un nodo con una sola clase presente tiene `Gini = 0`. Un nodo dividido 50/50 entre dos clases
tiene `Gini = 0.5`.
