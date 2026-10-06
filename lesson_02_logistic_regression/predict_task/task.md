# Implementar `predict`

`predict_proba` da una probabilidad. Para obtener una etiqueta de clase concreta, compárala con
`threshold` (por defecto `0.5`) y convierte el array booleano resultante a enteros (`0` o `1`):

```
y_pred = (predict_proba(X) >= threshold).astype(int)
```
