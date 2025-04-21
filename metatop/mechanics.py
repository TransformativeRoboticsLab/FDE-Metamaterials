import itertools

import fenics as fe
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy
from loguru import logger
from sympy.core.symbol import symbols

MANDEL = jnp.diag(jnp.array([1., 1., jnp.sqrt(2)]))
INV_MANDEL = jnp.diag(jnp.array([1., 1., 1./jnp.sqrt(2)]))


@jax.jit
def ray_q(A: jnp.ndarray, X: jnp.ndarray):
    """
    Calculate Rayleigh quotient(s) for vector(s) X with matrix M.

    Args:
        X: Vector (n,) or matrix (n,n) where columns are vectors
        A: Matrix (n,n) 

    Returns:
        If X is a vector: scalar Rayleigh quotient X.T @ A @ X
        If X is a matrix: array of Rayleigh quotients for each column
    """

    # Handle vector case
    if X.ndim == 1:
        return (X.T @ A @ X) / (X.T @ X)

    # Handle matrix case
    if X.ndim == 2:
        return jnp.diag(X.T @ A @ X) / jnp.diag(X.T @ X)

    raise ValueError("Input v must be 1D vector or 2D matrix")


@jax.jit
def mandelize(A):
    if A.shape != (3, 3):
        raise ValueError("Shape of input matrix is not 3x3")
    return MANDEL @ A @ MANDEL


@jax.jit
def inv_mandelize(A):
    if A.shape != (3, 3):
        raise ValueError("Shape of input matrix is not 3x3")
    return INV_MANDEL @ A @ INV_MANDEL


def linear_strain(u):
    return fe.sym(fe.grad(u))


def macro_strain(i: int) -> fe.Constant:
    eye = np.eye((3))[:, i]
    return fe.Constant(np.array([[eye[0],    eye[2]/2.],
                                 [eye[2]/2., eye[1]]]))


def lame_parameters(E, nu, model='plane_stress'):
    mu_ = E / (2.0 * (1.0 + nu))
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0*nu))
    if model == 'plane_stress':
        lambda_ = 2.0*mu_*lambda_ / (lambda_ + 2.0*mu_)

    return lambda_, mu_


def linear_stress(eps, E, nu):
    lambda_, mu_ = lame_parameters(E, nu, model='plane_stress')
    return lambda_*fe.tr(eps)*fe.Identity(2) + 2.0*mu_*eps


def tensor2matrix(C, output_style='mandel'):
    '''
    Converts a 2x2x2x2 elasticity tensor into a 3x3 elasticity matrix

    in: C, a 2x2x2x2 elasticity tensor
    out: V, a 3x3 elasticity matrix

    Reference for indexing: https://www.wikiwand.com/en/Linear_elasticity#Anisotropic_homogeneous_media
    '''
    lookup = {0: (0, 0), 1: (1, 1), 2: (0, 1)}
    V = np.zeros((3, 3), dtype=np.float64)
    for p in range(V.shape[0]):
        for q in range(V.shape[1]):
            i, j = lookup[p]
            k, l = lookup[q]
            V[p, q] = C[i, j, k, l]

    if output_style == 'mandel':
        a = np.diag(np.array([1, 1, np.sqrt(2)], dtype=np.double))
        V = a@V@a
    elif output_style == 'standard':
        pass
    else:
        raise ValueError('Incorrect output style')

    return V


def matrix2tensor(V, input_style='mandel'):
    """
    Original version using loops.
    Converts a 3x3 elasticity matrix into a 2x2x2x2 elasticity tensor.
    """
    if V.shape != (3, 3):
        raise ValueError("Input matrix V must be 3x3")
    if input_style not in ('standard', 'mandel'):
        raise ValueError(
            f"input_style must be 'mandel' or 'standard', not {input_style}")

    lookup = {(0, 0): 0, (1, 1): 1, (0, 1): 2, (1, 0): 2}
    # Ensure output dtype is float64 for calculations
    C = np.zeros((2, 2, 2, 2), dtype=np.float64)
    # Work with floats, copy to avoid modifying original V
    V_processed = V.astype(np.float64)

    if input_style == 'mandel':
        a = np.diag(np.array([1.0, 1.0, 1.0/np.sqrt(2.0)], dtype=np.float64))
        V_processed = np.matmul(a, np.matmul(V_processed, a))
    elif input_style == 'standard':
        pass  # No scaling needed

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            for k in range(C.shape[2]):
                for l in range(C.shape[3]):
                    p = lookup[(i, j)]
                    q = lookup[(k, l)]
                    C[i, j, k, l] = V_processed[p, q]
    return C


def matrix2tensor_vectorized(V, input_style='mandel'):
    """
    Vectorized version using advanced indexing (NumPy).
    Converts a 3x3 elasticity matrix into a 2x2x2x2 elasticity tensor.
    """
    if V.shape != (3, 3):
        raise ValueError("Input matrix V must be 3x3")
    if input_style not in ('standard', 'mandel'):
        raise ValueError(
            f"input_style must be 'mandel' or 'standard', not {input_style}")

    # Work with floats, ensure input V's dtype is preserved if float
    dtype_to_use = V.dtype if np.issubdtype(
        V.dtype, np.floating) else np.float64
    V_float = V.astype(dtype_to_use)

    if input_style == 'mandel':
        a = np.diag(np.array([1.0, 1.0, 1.0/np.sqrt(2.0)], dtype=dtype_to_use))
        V_scaled = np.matmul(a, np.matmul(V_float, a))
    else:
        V_scaled = V_float  # No scaling for 'standard'

    P = np.array([[0, 2],
                  [2, 1]], dtype=int)
    ii, jj, kk, ll = np.indices((2, 2, 2, 2))
    p_indices = P[ii, jj]
    q_indices = P[kk, ll]
    C = V_scaled[p_indices, q_indices]
    return C


def matrix2tensor_vectorized_jnp(V, input_style='mandel'):
    """
    Vectorized version using advanced indexing (JAX).
    Converts a 3x3 elasticity matrix into a 2x2x2x2 elasticity tensor.
    """
    if V.shape != (3, 3):
        raise ValueError("Input matrix V must be 3x3")
    if input_style not in ('standard', 'mandel'):
        raise ValueError(
            f"input_style must be 'mandel' or 'standard', not {input_style}")

    # JAX handles dtype promotion, ensure input is JAX array
    V_jnp = jnp.asarray(V)
    dtype_to_use = V_jnp.dtype

    if input_style == 'mandel':
        a = jnp.diag(
            jnp.array([1.0, 1.0, 1.0/jnp.sqrt(2.0)], dtype=dtype_to_use))
        # Use jnp.matmul or @
        V_scaled = a @ V_jnp @ a
    else:
        V_scaled = V_jnp  # No scaling for 'standard'

    P = jnp.array([[0, 2],
                   [2, 1]], dtype=int)
    ii, jj, kk, ll = jnp.indices((2, 2, 2, 2))
    p_indices = P[ii, jj]
    q_indices = P[kk, ll]
    C = V_scaled[p_indices, q_indices]
    return C


'''
So there's actually this weird bug in the matrix2tensor and calculate elastic constants functions. Originally I wrote matrix2tensor to handle elasticity matrices, specifically the C from \sigma = C \epsilon.

When C is in Voigt notation it is simply just a repackaging of the matrix values into the tensor locations.
However, this is not the case for S=C^-1 as seen on Hooke's Law wikipedia page
https://www.wikiwand.com/en/articles/Hooke's_law#Matrix_representation_(stiffness_tensor)
So in calculate elastic constants I invert C to S and then repackage it into a tensor, but not accounting for the scaling factors. 

This ended up hiding because it doesn't affect the Young's modulus but does affect the shear modulus.
Here I calculate G = 1 / S_1212, but that's not the usually equation. It is usually G = 1 / (2 * S_1212).
But because I don't account for the scaling factors when moving from S-matrix to S-tensor, then my calculation of G = 1 / S_1212 is actually correct.

Consequently, this is why I think I needed the Mandel fudge factor. Mandel notation is only a matrix thing, not a tensor thing so G = 1 / (2 * S_1212) should work whether we put in Mandel or Voigt; however, because I do the Mandel->Voigt conversion inside matrix2tensor I introduce a factor of 2 that needs to be pulled out.

So right now this is super cursed, but it's also super not worth rewriting. However, because I previously had a factor of 4 for my Mandel divisor, any G calculation I did with a mandel matrix is half of what it should be. 
'''


def calculate_elastic_constants(M, input_style='mandel'):
    """
    Original NumPy calculation using the loop-based matrix2tensor.
    """
    if input_style not in ('standard', 'mandel'):
        raise ValueError(
            f"input_style must be 'mandel' or 'standard', not {input_style}")

    # Uses original loop-based numpy version
    S_tensor = matrix2tensor(np.linalg.inv(M), input_style=input_style)

    # Ensure vectors are float64 to match S_tensor
    e1 = np.array([1.0, 0.0], dtype=np.float64)
    e2 = np.array([0.0, 1.0], dtype=np.float64)

    # Use np.einsum
    E1 = 1.0 / np.einsum('ijkl,i,j,k,l', S_tensor, e1, e1, e1, e1)
    E2 = 1.0 / np.einsum('ijkl,i,j,k,l', S_tensor, e2, e2, e2, e2)
    G12_denom = np.einsum('ijkl,i,j,k,l', S_tensor, e1, e2, e1, e2)
    G12 = 1.0 / G12_denom
    if input_style == 'mandel':
        G12 /= 2.0

    nu12 = -E1 * np.einsum('ijkl,i,j,k,l', S_tensor, e1, e1, e2, e2)
    nu21 = -E2 * np.einsum('ijkl,i,j,k,l', S_tensor, e2, e2, e1, e1)
    eta121 = E1 * np.einsum('ijkl,i,j,k,l', S_tensor, e1, e2, e1, e1)
    eta122 = E2 * np.einsum('ijkl,i,j,k,l', S_tensor, e1, e2, e2, e2)

    return {
        'E1': E1, 'E2': E2, 'G12': G12, 'nu12': nu12,
        'nu21': nu21, 'eta121': eta121, 'eta122': eta122
    }

# 5. JAX calculate_elastic_constants (einsum version, uses vectorized JAX helper)


def calculate_elastic_constants_jnp_einsum(M, input_style='mandel'):
    """
    JAX calculation using einsum and the vectorized JAX matrix2tensor.
    """
    if input_style not in ('standard', 'mandel'):
        raise ValueError(
            f"input_style must be 'mandel' or 'standard', not {input_style}")

    # Uses vectorized JAX version and jnp.linalg.inv
    S = matrix2tensor_vectorized_jnp(
        jnp.linalg.inv(M), input_style=input_style)

    e1 = jnp.array([1., 0.], dtype=S.dtype)
    e2 = jnp.array([0., 1.], dtype=S.dtype)

    E1 = 1.0 / jnp.einsum('ijkl,i,j,k,l', S, e1, e1,
                          e1, e1, optimize='optimal')
    E2 = 1.0 / jnp.einsum('ijkl,i,j,k,l', S, e2, e2,
                          e2, e2, optimize='optimal')
    G12_denom = jnp.einsum('ijkl,i,j,k,l', S, e1, e2,
                           e1, e2, optimize='optimal')
    G12 = 1.0 / G12_denom
    if input_style == 'mandel':
        G12 /= 2.0

    nu12 = -E1 * jnp.einsum('ijkl,i,j,k,l', S, e1, e1,
                            e2, e2, optimize='optimal')
    nu21 = -E2 * jnp.einsum('ijkl,i,j,k,l', S, e2, e2,
                            e1, e1, optimize='optimal')
    eta121 = E1 * jnp.einsum('ijkl,i,j,k,l', S, e1, e2,
                             e1, e1, optimize='optimal')
    eta122 = E2 * jnp.einsum('ijkl,i,j,k,l', S, e1, e2,
                             e2, e2, optimize='optimal')

    return {
        'E1': E1, 'E2': E2, 'G12': G12, 'nu12': nu12,
        'nu21': nu21, 'eta121': eta121, 'eta122': eta122
    }

# 6. JAX calculate_elastic_constants (no einsum version, uses vectorized JAX helper)


def calculate_elastic_constants_jnp_no_einsum(M, input_style='mandel'):
    """
    JAX calculation using manual indexing and the vectorized JAX matrix2tensor.
    """
    if input_style not in ('standard', 'mandel'):
        raise ValueError(
            f"input_style must be 'mandel' or 'standard', not {input_style}")

    # Uses vectorized JAX version and jnp.linalg.inv
    S = matrix2tensor_vectorized_jnp(
        jnp.linalg.inv(M), input_style=input_style)

    # Manual indexing
    E1 = 1.0 / S[0, 0, 0, 0]
    E2 = 1.0 / S[1, 1, 1, 1]
    # Assuming minor symmetries S_ijlk = S_ijkl and S_klij = S_ijkl for G12
    # S_1212 corresponds to i=0, j=1, k=0, l=1
    G12_denom = S[0, 1, 0, 1]
    G12 = 1.0 / G12_denom
    if input_style == 'mandel':
        G12 /= 2.0

    # Assuming major symmetry S_ijkl = S_klij for nu
    # nu12 is -E1 * S_1122 (i=0, j=0, k=1, l=1)
    nu12 = -E1 * S[0, 0, 1, 1]
    # nu21 is -E2 * S_2211 (i=1, j=1, k=0, l=0)
    nu21 = -E2 * S[1, 1, 0, 0]

    # eta are less standard, deriving indices from einsum version:
    # eta121 = E1 * S_1211 (i=0, j=1, k=0, l=0)
    eta121 = E1 * S[0, 1, 0, 0]
    # eta122 = E2 * S_1222 (i=0, j=1, k=1, l=1) - Wait, einsum was e1,e2,e2,e2 => S_1222
    eta122 = E2 * S[0, 1, 1, 1]  # Index corrected based on einsum

    return {
        'E1': E1, 'E2': E2, 'G12': G12, 'nu12': nu12,
        'nu21': nu21, 'eta121': eta121, 'eta122': eta122
    }


def anisotropy_index(C, input_style):
    '''
    Calculate the 2D anisotropy index of a given 2D 3x3 elasticity matrix

    in: C, a 3x3 elasticity matrix in standard (Voigt) notation
        input_style, can only be "standard" but is a check on the user that C is indeed in standard notation

    out: Kr, the Reuss estimate for bulk modulus
        Kv, the Voigt estimate for bulk modulus
        Gr, the Reuss estimate for shear modulus
        Gv, the Voigt estimate for shear modulus
        ASU, the anisotropy index

    references:
      * "Elastic anisotropy measure for two-dimensional crystals" by Li et al.
    '''

    if input_style == 'mandel':
        m = np.diag(np.array([1., 1., 1./np.sqrt(2)], dtype=np.double))
        C = m@C@m
    elif input_style == 'standard':
        pass
    else:
        raise ValueError('Incorrect input style')

    S = np.linalg.inv(C)
    Kr = 1/(S[0, 0]+S[1, 1]+2*S[0, 1])
    Gr = 2/(S[0, 0]+S[1, 1]-2*S[0, 1]+S[2, 2])
    Kv = (C[0, 0]+C[1, 1]+2*C[0, 1])/4
    Gv = (C[0, 0]+C[1, 1]-2*C[0, 1]+4*C[2, 2])/8
    ASU = np.sqrt((Kv/Kr - 1)**2 + 2*(Gv/Gr - 1)**2)

    return {
        'Kr': Kr,
        'Kv': Kv,
        'Gr': Gr,
        'Gv': Gv,
        'ASU': ASU
    }


def matrix_invariants(M):
    '''
    Calculate the invariants of a 3x3 matrix

    in: M, a 3x3 matrix

    out: I1, the first invariant
          I2, the second invariant
          I3, the third invariant
    '''
    I1 = np.trace(M)
    I2 = 0.5*(np.trace(M)**2 - np.trace(M@M))
    I3 = np.linalg.det(M)

    return {
        'tr(M)': I1,
        'dev(M)': I2,
        'det(M)': I3
    }


def convert_isotropic_properties(input_props: dict[str, float]):

    num_inputs = sum(1 for v in input_props.values() if v is not None)
    if num_inputs != 2:
        raise ValueError("Exactly two material properties must be provided.")

    E = input_props.get('E')
    G = input_props.get('G')
    nu = input_props.get('nu')
    K = input_props.get('K')

    for p, v in input_props.items():
        if p == 'nu':
            if v is not None and (v < -1 or v > 0.5):
                raise ValueError(
                    f"Poisson's ratio (nu = {nu} is outside the valid range [-1, 0.5])")
        else:
            if v is not None and v <= 0.:
                raise ValueError(f"Parameter {p} must be positive")

    try:
        if E is not None and G is not None:
            nu = E / (2 * G) - 1
            K = E*G / (3 * (3 * G - E))
        elif E is not None and nu is not None:
            G = E / (2 * (1 + nu))
            K = E / (3 * (1 - 2 * nu))
        elif E is not None and K is not None:
            nu = (3 * K - E) / (6 * K)
            G = (3 * K * E) / (9 * K - E)
        elif G is not None and nu is not None:
            E = 2 * G * (1 + nu)
            K = (2 * G * (1 + nu)) / (3 * (1 - 2 * nu))
        elif G is not None and K is not None:
            E = (9 * K * G) / (3 * K + G)
            nu = (3 * K - 2 * G) / (2 * (3 * K + G))
        elif nu is not None and K is not None:
            E = 3 * K * (1 - 2 * nu)
            G = (3 * K * (1 - 2 * nu)) / (2 * (1 + nu))

    except ZeroDivisionError as e:
        E = None
        G = None
        K = None
        nu = None
        logger.error(
            f"Encountered zero division when calculating properties: {e}")
    except Exception as e:
        logger.error(f"Unexpected error when calculating properties: {e}")

    return dict(E=E, G=G, K=K, nu=nu)


def isotropic_elasticity_matrix(E, nu, plane='stress', output_style='mandel'):
    """
    Compute the elasticity matrix for isotropic materials under plane stress or strain conditions.

    This function calculates the elasticity matrix based on the provided Young's modulus,
    Poisson's ratio, and the given condition of plane stress or strain. The output matrix
    can be in either 'mandel' or 'standard' format.

    Parameters
    ----------
    E : float
        Young's modulus of the material.
    nu : float
        Poisson's ratio of the material.
    plane : str, optional
        Condition of the material. It can be either 'stress' or 'strain'.
        Default is 'stress'.
    output_style : str, optional
        Output format of the elasticity matrix. It can be either 'mandel' or 'standard'.
        Default is 'mandel'.

    Returns
    -------
    C : numpy.ndarray
        The calculated elasticity matrix of the material under given conditions.

    Raises
    ------
    ValueError
        If the output_style is neither 'mandel' nor 'standard'.

    References
    ----------
    - https://www.wikiwand.com/en/Hooke%27s_law#Linear_elasticity_theory_for_continuous_media
    - https://www.wikiwand.com/en/Plane_stress
    - https://www.wikiwand.com/en/Hooke%27s_law#Isotropic_materials
    """
    if (plane == 'stress'):
        alpha = E / (1 - nu**2)
        C = np.array([[1, nu, 0],
                      [nu, 1, 0],
                      [0, 0, (1 - nu)/2]])
        C *= alpha
    elif (plane == 'strain'):
        alpha = E / (1 + nu) / (1 - 2 * nu)
        C = np.array([[1 - nu, nu, 0],
                      [nu, 1 - nu, 0],
                      [0, 0, (1-2*nu)/2]])
        C *= alpha

    if output_style == 'mandel':
        s = np.diag(np.array([1, 1, np.sqrt(2)]))
        C = s@C@s
    elif output_style == 'standard':
        pass
    else:
        raise ValueError('Incorrect output style')

    return C


def conversion_check():

    E0 = 1
    G0 = 0.384615
    K0 = 0.833333
    nu0 = 0.3
    props = {'E': E0, 'G': G0, 'K': K0, 'nu': nu0}
    keys = ['E', 'G', 'K', 'nu']

    for k1, k2 in itertools.combinations(keys, 2):
        input_props = {k: props[k] if k in [k1, k2] else None for k in keys}
        calculated_props = convert_isotropic_properties(input_props)
        # print(f"Input: {k1}={props[k1]}, {k2}={props[k2]}")
        # print(f"Calculated: {calculated_props}")
        for k, v in props.items():
            if not np.isclose(v, calculated_props[k]):
                raise ValueError(
                    f"Mismatch of calculated and input properties for {k}: input={v:.3f}, calcualted={calculated_props[k]}")
    logger.info("Passed property check")


if __name__ == "__main__":
    conversion_check()
