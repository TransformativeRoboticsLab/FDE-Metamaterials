import itertools

import fenics as fe
import jax
import jax.numpy as jnp
import numpy as np
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


def matrix2tensor(V, input_style='mandel'):
    '''
    Converts a 3x3 elasticity matrix into a 2x2x2x2 elasticity tensor

    in: V, a 3x3 elasticity matrix
        input_style, proper scaling for different matrix notation styles, e.g. Mandel or Voigt

    out: C, a 2x2x2x2 elasticity tensor

    Reference for indexing: https://www.wikiwand.com/en/Linear_elasticity#Anisotropic_homogeneous_media
    '''
    lookup = {(0, 0): 0, (1, 1): 1, (0, 1): 2, (1, 0): 2}
    # lookup3D = {(0,0): 0, (1,1): 1, (2,2): 2, (1,2): 3, (2,1): 3, (2,0): 4, (0,2): 4, (0,1): 5, (1,0): 5}
    C = np.zeros((2, 2, 2, 2), dtype=np.float64)

    if input_style == 'mandel':
        a = np.diag(np.array([1, 1, 1/np.sqrt(2)], dtype=np.double))
        V = a@V@a
    elif input_style == 'standard':
        pass
    else:
        raise ValueError('Incorrect input style')

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            for k in range(C.shape[2]):
                for l in range(C.shape[3]):
                    p = lookup[(i, j)]
                    q = lookup[(k, l)]
                    C[i, j, k, l] = V[p, q]

    return C


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


def calculate_elastic_constants(M, input_style='mandel'):
    '''
    Calculate the 2D elastic material properties of a 3x3 elasticity matrix

    in: M, a 3x3 elastic matrix
        input_style, input style of matrix, 'mandel' or 'standard'

    out: E1, Young's modulus in the direction of strain 1
         E2, Young's modulus in the direction of strain 2
         G12, Shear modulus of face 1 in the direction of strain 2
         nu12, Poisson ratio of transverse strain 2 direction when applying axial strain 1
         nu21, Poisson ratio of transverse strain 1 direction when applying axial strain 2
         eta121, Shear-normal coupling extension in strain 1 coupling applying shear strain
         eta122, Shear extension in strain 2 coupling applying shear strain

    Reference:
      1. "Extreme values of Young’s modulus and Poisson’s ratio of hexagonal crystals" by Gorodstov
      2. "Fundamentals of crystal physics" by Sirotin, p654

    Note: Ref 1 is where I found the equation, but they cite Ref 2 for the equations
          I also don't have validation for my equation for shear modulus,
              it was just based on intuition and confirming agains the 2*22 case,
              but it may be wrong for an oblique matrix???
    '''

    if input_style not in ('standard', 'mandel'):
        raise ValueError(
            f"input_style must be 'mandel' or 'standard', not {input_style}")

    S = matrix2tensor(np.linalg.inv(M), input_style=input_style)

    e1 = np.array([1, 0])
    e2 = np.array([0, 1])

    # all of this einstein summation is selecting components out of the S tensor
    E1 = 1/np.einsum('ijkl,i,j,k,l', S, e1, e1, e1, e1)  # 1/S_1111
    E2 = 1/np.einsum('ijkl,i,j,k,l', S, e2, e2, e2, e2)  # 1/S_2222
    G12 = 1/np.einsum('ijkl,i,j,k,l', S, e1, e2, e1, e2)  # 1/S_1212
    if input_style == 'mandel':
        G12 /= 4
    nu12 = -E1 * np.einsum('ijkl,i,j,k,l', S, e1, e1, e2, e2)
    nu21 = -E2 * np.einsum('ijkl,i,j,k,l', S, e2, e2, e1, e1)
    eta121 = E1 * np.einsum('ijkl,i,j,k,l', S, e1, e2, e1, e1)
    eta122 = E2 * np.einsum('ijkl,i,j,k,l', S, e1, e2, e2, e2)

    return {
        'E1': E1,
        'E2': E2,
        'G12': G12,
        'nu12': nu12,
        'nu21': nu21,
        'eta121': eta121,
        'eta122': eta122
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


if __name__ == "__main__":

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
