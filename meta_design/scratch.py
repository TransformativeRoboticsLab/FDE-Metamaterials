import jax
import jax.numpy as jnp
from jax.experimental import sparse
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from scipy.sparse import csr_matrix


# generate a nelx by nely mesh
nelx = 100
nely = nelx
rad = 0.5

mesh = np.random.uniform(0., 1., (nelx, nely))
# determine centers of each mesh if a side is unit length
print("Finding midpoints")
mids = np.array([[(i + 0.5)/nelx, (j + 0.5)/nely] for i in range(nelx) for j in range(nely)])

print("Finding distances")
distances = euclidean_distances(mids.reshape(-1,1))
# print(distances)

print("Calculating H")
H = np.maximum(0., rad - distances)
H[H < 1e-6] = 0.
print("Calculating Hs")
Hs = H.sum(axis=1)
print("Convering Hs to sparse")
H_sp = csr_matrix(H)

H_jax = sparse.BCOO.fromdense(H)
# print(H)