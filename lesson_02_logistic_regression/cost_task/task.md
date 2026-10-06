# Implementar `compute_cost`

Implementa el costo de entropía cruzada binaria:

```
cost = -(1 / n) * suma(y * log(p) + (1 - y) * log(1 - p))
```

Pasos:
1. Obtén `p = self.predict_proba(X)`.
2. **Recorta** (clip) `p` lejos de los límites exactos con `np.clip(p, eps, 1 - eps)`, usando un
   `eps` muy pequeño como `1e-15`. Sin esto, una predicción confiada y correcta (`p` extremadamente
   cerca de 0 o 1) puede hacer que `np.log` devuelva `-inf`, y el costo `nan`.
3. Devuelve `-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))`.

Pruébalo en la demo de `main.py`: el mismo modelo muy confiado obtiene un costo cercano a cero
cuando sus predicciones coinciden con `y`, y un costo grande cuando no coinciden.
