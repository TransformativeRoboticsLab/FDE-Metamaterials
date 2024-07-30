from fenics import sym, grad, Constant, tr, Identity
import numpy as np

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