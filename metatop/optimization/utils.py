import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from nlopt import ForcedStop

from metatop.mechanics import mandelize


@jax.jit
def process_C_unimode(C):
    M = mandelize(C)
    M /= jnp.linalg.norm(M, ord=2)
    return M


@jax.jit
def process_C_bimode(C):
    M = mandelize(C)
    M = jnp.linalg.inv(M)
    M /= jnp.linalg.norm(M, ord=2)
    return M


@jax.jit
def matrix_inv_sqrt(M):
    w, v = jnp.linalg.eigh(M)
    inv_sqrt_w = 1. / jnp.sqrt(w + 1e-6)
    return (v * inv_sqrt_w) @ v.T


@jax.jit
def matrix_log_spd(M):
    w, v = jnp.linalg.eigh(M)
    log_w = jnp.log(w + 1e-6)
    return (v * log_w) @ v.T


@jax.jit
def log_euclidean_distance_sq(A, B):
    logA = matrix_log_spd(A)
    logB = matrix_log_spd(B)
    diff = logA - logB
    return jnp.sum(jnp.square(diff))
    # return jnp.linalg.norm(diff)


@jax.jit
def affine_invariant_distance_sq(A, B):
    A_inv_sqrt = matrix_inv_sqrt(A)
    C = A_inv_sqrt @ B @ A_inv_sqrt
    return jnp.sum(jnp.square(matrix_log_spd(C)))
    # return jnp.linalg.norm(matrix_log_spd(C), ord='fro')


@jax.jit
def frobenius_distance_sq(A, B):
    diff = A - B
    return jnp.sum(jnp.square(diff))


@jax.jit
def commutation_distance_sq(A, B):
    diff = A @ B - B @ A
    return jnp.sum(jnp.square(diff))


def stop_on_nan(x: float | np.ndarray):
    if np.isnan(x).any():
        logger.error(
            "NaN value detected in objective function. Terminating optimization run.")
        raise ForcedStop
