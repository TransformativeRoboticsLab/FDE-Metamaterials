# import fenics as fe
from fenics import *
import numpy as np
import nlopt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import time

from metamaterial import Metamaterial
from filters import DensityFilter
from optimization import AndreassenOptimization, VolumeConstraint, IsotropicConstraint, BulkModulusConstraint, ShearModulusConstraint, OptimizationState, Epigraph, ExtremalConstraints
from helpers import Ellipse, print_summary, beta_function

RAND_SEED = 1

# when an epoch changes or we change beta the constraint values can jump
# and because the constraints can also be clamped by t we need to make sure
# that we start the epoch in a feasible state. 
# Basically t could be too low for the constraints to be satisfied and the 
# optimizer will spend cycles trying to get t up to a feasible value.
# We avoid this by jumping t to a feasible value at the start of each epoch.
def update_t(x, gs):
    print(f"Updating t...\nOld t value {x[-1]:.3e}")
    new_t = -np.inf
    x[-1] = 0.
    for g in gs:
        results = np.zeros(g.n_constraints)
        g(results, x, np.array([]), dummy_run=True)
        new_t = max(new_t, np.max(results)/g.eps)
    x[-1] = new_t
    print(f"New t value: {x[-1]:.3e}")

def main():
    nelx = 50
    nely = nelx
    E_max = 1.
    E_min = 1e-9
    nu = 0.3
    vol_frac = 0.5
    start_beta, n_betas = 1, 8
    betas = [start_beta * 2 ** i for i in range(n_betas)]
    # print(betas)
    eta = 0.5
    pen = 3.
    epoch_duration = 100
    a = 2e-3
    optim_type = 'npr' # shear, bulk, npr, ppr
    print_summary(optim_type, nelx, nely, E_max, E_min, nu, vol_frac, betas, eta, pen, epoch_duration, a)
    
    metamate = Metamaterial(E_max, E_min, nu)
    metamate.mesh = UnitSquareMesh(nelx, nely, 'crossed')
    # metamate.mesh = RectangleMesh.create([Point(0, 0), Point(1, 1)], [nelx, nely], CellType.Type.quadrilateral)
    metamate.create_function_spaces()
    
    filt = DensityFilter(metamate.mesh, 0.1, distance_method='periodic')
    
    ops = OptimizationState()
    ops.beta = start_beta
    ops.eta = eta
    ops.pen = pen
    ops.filt = filt

    np.random.seed(RAND_SEED)
    # x = np.random.uniform(0, 1, dim)
    # x = beta_function(vol_frac, metamate.R.dim())
    # x = np.random.binomial(1, vol_frac, metamate.R.dim())
    x = np.random.choice([0., 1.], metamate.R.dim())
    x = np.append(x, 1.)
    metamate.x.vector()[:] = x[:-1]
    metamate.plot_density()
    # r = Function(metamate.R)
    # r.assign(interpolate(Ellipse(vol_frac, 1/3, 1/6), metamate.R))
    # x = r.vector()[:]
    
    v = np.eye(3)
    f = Epigraph()
    g = ExtremalConstraints(v=v, extremal_mode=1, metamaterial=metamate, ops=ops)

    opt = nlopt.opt(nlopt.LD_MMA, x.size)
    opt.set_min_objective(f)
    opt.add_inequality_mconstraint(g, np.zeros(g.n_constraints))
    
    lb = np.zeros(x.size)
    lb[-1] = 1e-6
    opt.set_lower_bounds(lb)
    ub = np.ones(x.size)
    ub[-1] = np.inf
    opt.set_upper_bounds(ub)
    opt.set_maxeval(50)
    # opt.set_param('inner_maxeval', 1_000)
    # opt.set_param('dual_maxeval',  1_000)

    # progressively up the projection
    for beta in betas:
        ops.beta = beta
        ops.epoch += 1
        update_t(x, [g])
        x_opt = opt.optimize(x)
        
        x = np.copy(x_opt)
        
        opt.set_maxeval(epoch_duration)

    plt.show(block=True)


if __name__ == "__main__":
    np.random.seed(1)
    main()