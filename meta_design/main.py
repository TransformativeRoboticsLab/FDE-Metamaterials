import fenics as fe
import numpy as np
import nlopt
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from metamaterial import Metamaterial
from filters import DensityFilter
from optimization import Objective, VolumeConstraint
from helpers import Circle

def main():
    nelx = 50
    nely = nelx
    E_max = 1.
    E_min = 1e-9
    nu = 0.3
    vol_frac = 0.5
    beta = 16
    eta = 0.5
    penalty = 3.
    
    metamate = Metamaterial(E_max, E_min, nu)
    metamate.mesh = fe.UnitSquareMesh(nelx, nely, 'crossed')
    metamate.create_function_spaces()
    
    filt = DensityFilter(metamate.mesh, 0.1, distance_method='periodic')

    dim = metamate.R.dim()
    # x = jnp.array(np.random.binomial(1, vol_frac, dim), dtype=np.float32)
    r = fe.Function(metamate.R)
    r.interpolate(Circle(V=vol_frac, rad=0.33))
    x = r.vector()[:]
    
    
    f = Objective(metamaterial=metamate, filt=filt, beta=beta, eta=eta, pen=penalty)
    g = VolumeConstraint(V=vol_frac, filt=filt, beta=beta, eta=eta)

    opt = nlopt.opt(nlopt.LD_MMA, dim)
    opt.set_min_objective(f)
    opt.add_inequality_constraint(g, 0.)
    opt.set_lower_bounds(0.)
    opt.set_upper_bounds(1.)
    opt.set_maxeval(100)
    
    opt.optimize(x)
    
    
    
    # sols, Chom = metamate.solve()
    # print(np.array(Chom))
    
    # baseChom, baseUChom = metamate.homogenized_C(sols, E_max, nu)
    # dd = fe.derivative(baseUChom[2][2], metamate.x)
    # print(np.array(baseChom))
    
    
    
    
    # return
    
    # pr = PoissonRatio(metamaterial=metamate, filt=filt, beta=1, eta=0.5)
    
    # pr(x, [])
    
    # S = np.array([[1., -nu, 0],
    #               [-nu, 1., 0.],
    #               [0., 0., 2*(1+nu)]]) / vol_frac
    
    # C = np.linalg.inv(S)
    # print(C)
    
    
    # the flow of the program
    # what should the metamaterial do?
    # what should the optimizer do?
    
    
    # initial rho
    # x = fe.Function(R)
    # x.vector()[:] = np.random.binomial(1, vol_frac, R.dim())
    
    # metamate.x = filt.filter(x)
    
    # sols, Chom = metamate.solve()
    
    
    # def objective(x, grad):
        


if __name__ == "__main__":
    np.random.seed(1)
    main()
