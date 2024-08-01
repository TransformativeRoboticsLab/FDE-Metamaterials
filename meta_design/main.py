import fenics as fe
import numpy as np
import nlopt
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from metamaterial import Metamaterial
from filters import DensityFilter
from optimization import Objective, VolumeConstraint, XiaOptimization
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
    # x = jnp.array(np.random.binomial(1, vol_frac, dim), dtype=np.float32)
    x = np.random.uniform(0, 1, dim)
    # r = fe.Function(metamate.R)
    # r.interpolate(Circle(V=vol_frac, rad=1/3))
    # x = r.vector()[:]

    # opt = XiaOptimization(metamaterial=metamate, filt=filt, pen=penalty, vol_frac=vol_frac, max_eval=max_eval)
    
    # x = opt.optimize(x)
    
    # metamate.x.vector()[:] = x
    # metamate.plot_density()
    
    
    f = Objective(metamaterial=metamate, filt=filt, beta=beta, eta=eta, pen=pen)
    g_vol = VolumeConstraint(V=vol_frac, filt=filt, beta=beta, eta=eta)
    # g_iso = IsotropicConstraint(filt=filt, beta=beta, eta=eta)

    opt = nlopt.opt(nlopt.LD_MMA, dim)
    opt.set_min_objective(f)
    opt.add_inequality_constraint(g_vol, 0.)
    opt.set_lower_bounds(0.)
    opt.set_upper_bounds(1.)
    opt.set_maxeval(max_eval)
    
    opt.optimize(x)


if __name__ == "__main__":
    np.random.seed(1)
    main()
