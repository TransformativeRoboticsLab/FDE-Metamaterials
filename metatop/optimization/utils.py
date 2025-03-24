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
def logm(M):
    w, v = jnp.linalg.eigh(M)
    log_w = jnp.log(jnp.clip(w, min=1e-8))
    return (v * log_w) @ v.T


@jax.jit
def sqrtm(M):
    w, v = jnp.linalg.eigh(M)
    sqrt_w = jnp.sqrt(w)
    return (v * sqrt_w) @ v.T


@jax.jit
def airm_dist(A, B):
    """ Affine Invariant Riemannian Metric Distance """
    A_sqrt_inv = sqrtm(jnp.linalg.inv(A))
    C = A_sqrt_inv @ B @ A_sqrt_inv
    return jnp.linalg.norm(logm(C), ord='fro')


@jax.jit
def log_euclidean_dist(A, B):
    logA = logm(A)
    logB = logm(B)
    diff = logA - logB
    # return jnp.sum(jnp.square(diff))
    return jnp.linalg.norm(diff, ord='fro')


@jax.jit
def frobenius_dist(A, B):
    diff = A - B
    # return jnp.sum(jnp.square(diff))
    return jnp.linalg.norm(diff, ord='fro')


@jax.jit
def sqrtm_dist(A, B):
    Asqrt = sqrtm(A)
    Bsqrt = sqrtm(B)
    diff = Asqrt - Bsqrt
    # return jnp.sum(jnp.square(Asqrt-Bsqrt))
    return jnp.linalg.norm(diff, ord='fro')


@jax.jit
def eigenpair_dist(A, B):
    # This doesn't really work when we're always normalizing the matrices
    wA, vA = jnp.linalg.eigh(A)
    wB, vB = jnp.linalg.eigh(B)

    # Calculate similarity matrix between eigenvectors of A and B
    similarity = jnp.abs(vA.T @ vB)

    # Find the best match in A for each eigenvector in B
    best_matches = jnp.argmax(similarity, axis=0)

    # Reorder eigenvalues and eigenvectors of A to align with B
    wA_aligned = wA[best_matches]
    vA_aligned = vA[:, best_matches]

    eigval_pen = jnp.sum(jnp.square(wA_aligned - wB))
    eigvec_pen = jnp.sum(1 - jnp.abs(jnp.diag(vA_aligned.T @ vB)))

    return eigval_pen + eigvec_pen


@jax.jit
def outer_product_cols(A):
    outer_prods = []
    for i in range(A.shape[0]):
        outer_prods.append(jnp.outer(A[:, 0], A[:, 0]))
    return outer_prods


@jax.jit
def commutation_distance_sq(A, B):
    diff = A @ B - B @ A
    return jnp.sum(jnp.square(diff))


@jax.jit
def principal_angle_loss(A, B):
    cos_angles = jnp.linalg.svd(A.T @ B, compute_uv=False)
    return jnp.sum(jnp.square(1 - cos_angles))


def stop_on_nan(x: float | np.ndarray):
    if np.isnan(x).any():
        logger.error(
            "NaN value detected in objective function. Terminating optimization run.")
        raise ForcedStop
