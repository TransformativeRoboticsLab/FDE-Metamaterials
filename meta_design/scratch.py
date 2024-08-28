import numpy as np
from jax import numpy as jnp
import jax
from matplotlib import pyplot as plt

np.random.seed(0)
A = np.random.uniform(0,1,(3,3))
A = (A @ A.T) / 2

w,v = np.linalg.eig(A)

lambda_max = np.max(w)
est_lambda_max = np.log(np.linalg.det(np.eye(3) + A))
tr_lambda_max = np.trace(A)

print(f"lambda_max: {lambda_max:.3f}")
print(f"est_lambda_max: {est_lambda_max:.3f}")
print(f"tr_lambda_max: {tr_lambda_max:.3f}")