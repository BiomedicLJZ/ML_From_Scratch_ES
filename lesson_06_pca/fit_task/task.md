# Implementar `fit`

Implementa `fit(self, X)`, que aprende los componentes principales a partir de los datos de
entrenamiento:

1. `self.mean_ = X.mean(axis=0)`.
2. Centra los datos: `X_centered = self._center(X)` (ya implementado).
3. Calcula la matriz de covarianza: `cov = np.cov(X_centered, rowvar=False)`.
4. Descompón en autovalores/autovectores. Como una matriz de covarianza es simétrica, usa
   `eigenvalues, eigenvectors = np.linalg.eigh(cov)` — `eigh` devuelve los autovalores en orden
   **ascendente**, y `eigenvectors[:, i]` es el autovector de `eigenvalues[i]`.
5. Ordena ambos de forma descendente: `order = np.argsort(eigenvalues)[::-1]`, luego reordena
   ambos arrays con `order`.
6. Quédate solo con los primeros `self.n_components`: guárdalos como `self.components_ =
   eigenvectors[:, :self.n_components].T` (de modo que cada fila de `components_` sea una
   dirección principal) y `self.explained_variance_ = eigenvalues[:self.n_components]`.
7. Devuelve `self`.

Recuerda: el signo de cada autovector es arbitrario. Tus componentes pueden apuntar en la
dirección opuesta a los de la implementación correcta de otra persona — eso está bien.
