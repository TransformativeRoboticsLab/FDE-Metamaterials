import fenics as fe
import jax

jax.config.update("jax_enable_x64", True)
import nlopt
import numpy as np
from matplotlib import pyplot as plt

from metatop.filters import DensityFilter
from metatop.helpers import beta_function, print_summary
from metatop.metamaterial import Metamaterial
from metatop.optimization import (AndreassenOptimization,
                                  BulkModulusConstraint, IsotropicConstraint,
                                  OptimizationState, ShearModulusConstraint,
                                  VolumeConstraint)

RAND_SEED = 0

def main():
    nelx = 50
    nely = nelx
    E_max = 1.
    E_min = 1e-9
    nu = 0.3
    vol_frac = 0.35
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
    # metamate.mesh = UnitSquareMesh(nelx, nely, 'crossed')
    metamate.mesh = fe.RectangleMesh.create([fe.Point(0, 0), fe.Point(1, 1)], [nelx, nely], fe.CellType.Type.quadrilateral)
    metamate.create_function_spaces()
    
    filt = DensityFilter(metamate.mesh, 0.1, distance_method='periodic')
    
    ops = OptimizationState()
    ops.beta = start_beta
    ops.eta = eta
    ops.pen = pen
    ops.filt = filt

    dim = metamate.R.dim()
    np.random.seed(RAND_SEED)
    # x = np.random.uniform(0, 1, dim)
    x = beta_function(vol_frac, dim)
    # x = np.random.binomial(1, vol_frac*0.95, dim)
    # r = fe.Function(metamate.R)
    # r.assign(interpolate(Ellipse(vol_frac, 1/3, 1/6), metamate.R))
    # x = r.vector()[:]
    
    f = AndreassenOptimization(optim_type=optim_type, metamaterial=metamate, ops=ops, plot=True, filter_and_project=True)
    g_vol = VolumeConstraint(V=vol_frac, ops=ops)
    g_iso = IsotropicConstraint(eps=1e-5, ops=ops)
    g_blk = BulkModulusConstraint(E_max, nu, a=a, ops=ops) # a = 0.02%
    g_shr = ShearModulusConstraint(E_max, nu, a=a, ops=ops)

    opt = nlopt.opt(nlopt.LD_MMA, dim)
    opt.set_min_objective(f)
    opt.add_inequality_constraint(g_vol, 0.)
    opt.add_inequality_constraint(g_iso, 0.)
    if optim_type in ['bulk', 'ppr']:
        opt.add_inequality_constraint(g_shr, 0.)
    elif optim_type in ['shear', 'npr']:
        opt.add_inequality_constraint(g_blk, 0.)

    opt.set_lower_bounds(0.)
    opt.set_upper_bounds(1.)
    opt.set_maxeval(2000)

    # progressively up the projection
    for beta in betas:
        ops.beta = beta
        f.epoch += 1
        x_opt = opt.optimize(x)
        
        x = x_opt
        
        opt.set_maxeval(epoch_duration)

    plt.show(block=True)


if __name__ == "__main__":
    np.random.seed(1)
    main()