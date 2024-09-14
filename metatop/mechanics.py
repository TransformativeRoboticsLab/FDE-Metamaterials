# from fenics import sym, grad, Constant, tr, Identity
import numpy as np
from fenics import *


def linear_strain(u):
    return sym(grad(u))

def macro_strain(i: int) -> Constant:
    eye = np.eye((3))[:, i]
    return Constant(np.array([[eye[0],    eye[2]/2.],
                              [eye[2]/2., eye[1]]]))

def lame_parameters(E, nu, model='plane_stress'):
    mu_ = E / (2.0 * (1.0 + nu))
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0*nu))
    if model == 'plane_stress':
        lambda_ = 2.0*mu_*lambda_ / (lambda_ + 2.0*mu_)

    return lambda_, mu_

def linear_stress(eps, E, nu):
    lambda_, mu_ = lame_parameters(E, nu, model='plane_stress')
    return lambda_*tr(eps)*Identity(2) + 2.0*mu_*eps

def matrix2tensor(V, input_style='mandel'):
  '''
  Converts a 3x3 elasticity matrix into a 2x2x2x2 elasticity tensor

  in: V, a 3x3 elasticity matrix
      input_style, proper scaling for different matrix notation styles, e.g. Mandel or Voigt

  out: C, a 2x2x2x2 elasticity tensor

  Reference for indexing: https://www.wikiwand.com/en/Linear_elasticity#Anisotropic_homogeneous_media
  '''
  lookup = {(0,0): 0, (1,1): 1, (0,1): 2, (1,0): 2}
  # lookup3D = {(0,0): 0, (1,1): 1, (2,2): 2, (1,2): 3, (2,1): 3, (2,0): 4, (0,2): 4, (0,1): 5, (1,0): 5}
  C = np.zeros((2,2,2,2), dtype=np.float64)

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
          p = lookup[(i,j)]
          q = lookup[(k,l)]
          C[i,j,k,l] = V[p,q]

  return C

def tensor2matrix(C, output_style='mandel'):
  '''
  Converts a 2x2x2x2 elasticity tensor into a 3x3 elasticity matrix

  in: C, a 2x2x2x2 elasticity tensor
  out: V, a 3x3 elasticity matrix

  Reference for indexing: https://www.wikiwand.com/en/Linear_elasticity#Anisotropic_homogeneous_media
  '''
  lookup = {0: (0, 0), 1: (1, 1), 2: (0, 1)}
  V = np.zeros((3,3), dtype=np.float64)
  for p in range(V.shape[0]):
    for q in range(V.shape[1]):
      i,j = lookup[p]
      k,l = lookup[q]
      V[p,q] = C[i,j,k,l]

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
       nu12, Poisson ratio in strain 2 direction when applying normal strain 1
       nu21, Poisson ratio in strain 1 direction when applying normal strain 2
       eta121, Shear extension in strain 1 coupling applying shear strain
       eta122, Shear extension in strain 2 coupling applying shear strain

  Reference:
    1. "Extreme values of Young’s modulus and Poisson’s ratio of hexagonal crystals" by Gorodstov
    2. "Fundamentals of crystal physics" by Sirotin, p654

  Note: Ref 1 is where I found the equation, but they cite Ref 2 for the equations
        I also don't have validation for my equation for shear modulus,
            it was just based on intuition and confirming agains the 2*22 case,
            but it may be wrong for an oblique matrix???
  '''
  S = matrix2tensor(np.linalg.inv(M), input_style=input_style)

  e1 = np.array([1, 0])
  e2 = np.array([0, 1])

  E1   = 1/np.einsum('ijkl,i,j,k,l',S,e1,e1,e1,e1)
  E2   = 1/np.einsum('ijkl,i,j,k,l',S,e2,e2,e2,e2)
  G12  = 1/np.einsum('ijkl,i,j,k,l',S,e1,e2,e1,e2)
  if input_style == 'mandel':
    G12 /= 4
  nu12   = -E1 * np.einsum('ijkl,i,j,k,l',S,e1,e1,e2,e2)
  nu21   = -E2 * np.einsum('ijkl,i,j,k,l',S,e2,e2,e1,e1)
  eta121 =  E1 * np.einsum('ijkl,i,j,k,l',S,e1,e2,e1,e1)
  eta122 =  E2 * np.einsum('ijkl,i,j,k,l',S,e1,e2,e2,e2)

  return E1, E2, G12, nu12, nu21, eta121, eta122

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

  if input_style != 'standard':
    raise ValueError('Only valid input style is "standard"')

  S = np.linalg.inv(C)
  Kr = 1/(S[0,0]+S[1,1]+2*S[0,1])
  Gr = 2/(S[0,0]+S[1,1]-2*S[0,1]+S[2,2])
  Kv = (C[0,0]+C[1,1]+2*C[0,1])/4
  Gv = (C[0,0]+C[1,1]-2*C[0,1]+4*C[2,2])/8
  ASU = np.sqrt((Kv/Kr - 1)**2 + 2*(Gv/Gr - 1)**2)

  return Kr, Kv, Gr, Gv, ASU