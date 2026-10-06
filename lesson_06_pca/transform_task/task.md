# Implementar `transform`

Implementa `transform(self, X)`, que proyecta datos nuevos sobre los componentes aprendidos por
`fit`:

1. Centra `X` usando la media guardada durante `fit`: `self._center(X)` (ya implementado).
2. Proyecta sobre los componentes: `.dot(self.components_.T)`.

Devuelve el resultado — un array `(n_samples, n_components)`, la versión de menor dimensión de
`X`.

Una vez que esto funcione tendrás una implementación completa de PCA: `fit` aprende las
direcciones de máxima varianza, y `transform` proyecta cualquier dato sobre ellas.
