import sympy as sp
from sympy import symbols, separatevars as sv, collect as cl
from sympy.matrices import Matrix
sp.init_printing(use_unicode=True)

# This script does a simple derivation of the plane stress equations from the 3D stress-strain equations
# It starts by building an isotropic stiffness matrix and then solves for the zero terms in the 3D stress-strain equations
# Those solved terms are then substituted back into the 3D stress-strain equations to get the plane stress equations
# The result is the coefficients fo the 2D material stiffness matrix in terms of the 3D material stiffness matrix
epsilon_xx, epsilon_yy, epsilon_zz, epsilon_xz, epsilon_yz, epsilon_xy = symbols('epsilon_xx epsilon_yy epsilon_zz epsilon_xz epsilon_yz epsilon_xy')
epsilon = Matrix([epsilon_xx, epsilon_yy, epsilon_zz, epsilon_xz, epsilon_yz, epsilon_xy])

# Symmetric, isotropic compliance matrix
C = Matrix(6, 6, lambda i, j: symbols(f'C{i+1}{j+1}' if i <= j else f'C{j+1}{i+1}'))
C[3:, 0:3] = sp.zeros(3, 3)
C[0:3, 3:] = sp.zeros(3, 3)
C[3, 4:] = sp.zeros(1, 2)
C[4:, 3] = sp.zeros(2, 1)
C[4, 5] = 0
C[5, 4] = 0

# establish sigma, solve for the zero terms and substitute back in
sigma = C*epsilon
sigma_solves = sp.solve([sigma[2], sigma[3], sigma[4]], [epsilon_zz, epsilon_xz, epsilon_yz])
sigma_substituted = sigma.subs(sigma_solves)

# collect coefficients
eps_2d = (epsilon_xx, epsilon_yy, epsilon_xy)
sigma_xx, sigma_yy, sigma_xy = sigma_substituted[0], sigma_substituted[1], sigma_substituted[5]
coeffs_xx = [cl(sigma_xx.expand(), eps_2d).coeff(var) for var in (epsilon_xx, epsilon_yy, epsilon_xy)]
coeffs_yy = [cl(sigma_yy.expand(), eps_2d).coeff(var) for var in (epsilon_xx, epsilon_yy, epsilon_xy)]
coeffs_xy = [cl(sigma_xy.expand(), eps_2d).coeff(var) for var in (epsilon_xx, epsilon_yy, epsilon_xy)]

plane_stress_matrix = sp.Matrix([[coeffs_xx[0], coeffs_xx[1], coeffs_xx[2]],
                                 [coeffs_yy[0], coeffs_yy[1], coeffs_yy[2]],
                                 [coeffs_xy[0], coeffs_xy[1], coeffs_xy[2]]])

print("3x3 Plane Stress Matrix:")
sp.pprint(plane_stress_matrix)

# K*e_v = -p
# K*(e_11 + e_22)/2 = -p
# K = -2*p / (e_11 + e_22)

