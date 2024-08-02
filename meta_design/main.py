import fenics as fe
import numpy as np
import nlopt
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import time

from metamaterial import Metamaterial
from filters import DensityFilter
from optimization import Objective, VolumeConstraint, IsotropicConstraint
from helpers import Circle

def main():
    nelx = 50
    nely = nelx
    E_max = 1.
    E_min = 1e-9
    nu = 0.3
    vol_frac = 0.35
    beta = 1
    eta = 0.5
    pen = 3.
    max_eval = 200
    
    metamate = Metamaterial(E_max, E_min, nu)
    metamate.mesh = fe.UnitSquareMesh(nelx, nely, 'crossed')
    metamate.create_function_spaces()
    
    filt = DensityFilter(metamate.mesh, 0.1, distance_method='periodic')

    dim = metamate.R.dim()
    x = np.random.uniform(0, 1, dim)
    
    f = Objective(metamaterial=metamate, filt=filt, beta=beta, eta=eta, pen=pen, plot=False)
    g_vol = VolumeConstraint(V=vol_frac, filt=filt, beta=beta, eta=eta)
    g_iso = IsotropicConstraint()

    f(x, np.array([]))
    g_iso(x, np.array([]))

    # opt = nlopt.opt(nlopt.LD_MMA, dim)
    # opt.set_min_objective(f)
    # opt.add_inequality_constraint(g_vol, 0.)
    # opt.set_lower_bounds(0.)
    # opt.set_upper_bounds(1.)
    # opt.set_maxeval(max_eval)
    
    # start_time = time.time()
    # opt.optimize(x)
    # end_time = time.time()
    # duration = end_time - start_time
    # print(f"Optimization took {duration} seconds")


if __name__ == "__main__":
    np.random.seed(1)
    main()