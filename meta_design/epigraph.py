# import fenics as fe
from helpers import Ellipse, print_summary, beta_function
from optimization import OptimizationState, Epigraph, ExtremalConstraints, MaterialSymmetryConstraints, OffDiagonalConstraint, EpigraphBulkModulusConstraint, EigenvectorConstraint, InvariantsConstraint
from filters import DensityFilter
from metamaterial import Metamaterial
from mechanics import calculate_elastic_constants, anisotropy_index
import time
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import jax.numpy as jnp
from fenics import *
import numpy as np
import nlopt
import jax
jax.config.update("jax_enable_x64", True)


np.set_printoptions(precision=5)
# np.set_printoptions(suppress=True)


def uniform_density(dim):
    return np.random.uniform(0., 1., dim)


def beta_density(vol_frac, dim):
    return beta_function(vol_frac, dim)


def binomial_density(vol_frac, dim):
    return np.random.binomial(1, vol_frac, dim)


density_functions = {
    'uniform': lambda dim: uniform_density(dim),
    'beta': lambda vol_frac, dim: beta_density(vol_frac, dim),
    'binomial': lambda vol_frac, dim: binomial_density(vol_frac, dim)
}


def init_density(density_seed_type, vol_frac, dim):
    if density_seed_type not in density_functions:
        raise ValueError(f"Invalid density_seed_type: {density_seed_type}")
    return density_functions[density_seed_type](vol_frac, dim) if density_seed_type != 'uniform' else density_functions[density_seed_type](dim)


RAND_SEED = 1
print(f"Random Seed: {RAND_SEED}")
np.random.seed(RAND_SEED)
nlopt.srand(RAND_SEED)

ISQR2 = 1. / np.sqrt(2.)
v_dict = {
    "BULK": np.array([[ISQR2, -ISQR2, 0.],
                      [ISQR2,  ISQR2, 0.],
                      [0.,     0.,    1.]]),
    "IBULK": np.array([[-ISQR2, ISQR2, 0.],
                       [ISQR2, ISQR2, 0.],
                       [0.,    0.,    1.]]),
    "VERT": np.array([[0., 1., 0.],
                      [1., 0., 0.],
                      [0., 0., 1.]]),
    "VERT2": np.array([[0., ISQR2, -ISQR2],
                       [1., 0.,     0.],
                       [0., ISQR2,  ISQR2]]),
    "SHEAR": np.array([[0., -ISQR2, ISQR2],
                       [0.,  ISQR2, ISQR2],
                       [1.,  0.,    0.]]),
    "SHEARXY": np.array([[0., 1., 0.],
                         [0., 0., 1.],
                         [1., 0., 0.]]),
    "HSA": np.array([[0.,     0.,    1.],
                    [ISQR2, -ISQR2, 0.],
                    [ISQR2,  ISQR2, 0.]]),
    "HSA2": np.array([[0.,     0.,    1.],
                     [ISQR2,  ISQR2, 0.],
                      [-ISQR2, ISQR2, 0.]]),
    "IHSA": np.array([[1., 0., 0.],
                      [0., ISQR2, -ISQR2],
                      [0., ISQR2,  ISQR2]]),
    "EYE": np.eye(3),
}

# when an epoch changes or we change beta the constraint values can jump
# and because the constraints can also be clamped by t we need to make sure
# that we start the epoch in a feasible state.
# Basically t could be too low for the constraints to be satisfied and the
# optimizer will spend cycles trying to get t up to a feasible value.
# We avoid this by jumping t to a feasible value at the start of each epoch


def update_t(x, gs):
    print(f"Updating t...\nOld t value {x[-1]:.3e}")
    new_t = -np.inf
    x[-1] = 0.
    for g in gs:
        results = np.zeros(g.n_constraints)
        g(results, x, np.array([]), dummy_run=True)
        new_t = max(new_t, *(results/g.eps))
    x[-1] = new_t
    print(f"New t value: {x[-1]:.3e}")


def setup_metamaterial(E_max, E_min, nu, nelx, nely, mesh_cell_type='triangle'):
    metamaterial = Metamaterial(E_max, E_min, nu)
    if 'tri' in mesh_cell_type:
        metamaterial.mesh = UnitSquareMesh(nelx, nely, 'crossed')
    elif 'quad' in mesh_cell_type:
        metamaterial.mesh = RectangleMesh.create([Point(0, 0), Point(1, 1)], [
                                                 nelx, nely], CellType.Type.quadrilateral)
    else:
        raise ValueError(f"Invalid cell_type: {mesh_cell_type}")
    metamaterial.create_function_spaces()
    return metamaterial


def main():
    # inputs
    nelx = 50
    nely = nelx
    E_max = 1.
    E_min = 1e-9
    nu = 0.3
    vol_frac = 0.1
    start_beta, n_betas = 8, 3
    betas = [start_beta * 2 ** i for i in range(n_betas)]
    # betas.append(betas[-1]) # repeat the last beta for final epoch when we turn on constraints
    print(f"Betas: {betas}")
    eta = 0.5
    epoch_duration = 100
    basis_v = 'BULK'
    density_seed_type = 'uniform'
    extremal_mode = 1
    mesh_cell_type = 'tri'  # triangle, quadrilateral

    metamate = setup_metamaterial(
        E_max, E_min, nu, nelx, nely, mesh_cell_type=mesh_cell_type)

    # density filter setup
    filt = DensityFilter(metamate.mesh, 0.1, distance_method='periodic')

    # global optimization state
    ops = OptimizationState(beta=start_beta, eta=eta,
                            filt=filt, epoch_iter_tracker=[1])

    # seeding the initial density
    x = init_density(density_seed_type, vol_frac, metamate.R.dim())
    x = np.append(x, 0.)  # append t value for epigraph form

    # setup optimization
    v = v_dict[basis_v]
    f = Epigraph()
    g_ext = ExtremalConstraints(
        v=v, extremal_mode=extremal_mode, metamaterial=metamate, ops=ops, plot_interval=10)
    g_inv = InvariantsConstraint(ops=ops, verbose=True)
    g_vec = EigenvectorConstraint(v=v, ops=ops, eps=1e-1, verbose=True)
    active_constraints = [g_ext, ]

    opt = nlopt.opt(nlopt.LD_MMA, x.size)
    opt.set_min_objective(f)
    for g in active_constraints:
        opt.add_inequality_mconstraint(g, np.zeros(g.n_constraints))

    # opt.add_inequality_mconstraint(g_inv, np.zeros(g_inv.n_constraints))

    opt.set_lower_bounds(np.append(np.zeros(x.size - 1), 1e-10))
    opt.set_upper_bounds(np.append(np.ones(x.size - 1), np.inf))
    opt.set_maxeval(epoch_duration)
    opt.set_param('dual_ftol_rel', 1e-6)

    # progressively up the projection
    for n, beta in enumerate(betas, 1):
        ops.beta, ops.epoch = beta, n
        update_t(x, active_constraints)
        x = np.copy(opt.optimize(x))
        ops.epoch_iter_tracker.append(len(g_ext.evals))

        g_vec.eps = np.maximum(1e-3, g_vec.eps / 2.)
        opt.set_maxeval(epoch_duration)

    metamate.x.vector()[:] = x[:-1]
    final_C = np.asarray(metamate.solve()[1])
    print('Final C:\n', final_C)
    w, v = np.linalg.eigh(final_C)
    print('Final Eigenvalues:\n', w)
    print('Final Eigenvalue Ratios:\n', w / np.max(w))
    print('Final Eigenvectors:\n', v)

    print('Final ASU:', anisotropy_index(final_C, input_style='standard')[-1])
    print('Final Elastic Constants:', calculate_elastic_constants(
        final_C, input_style='standard'))

    plt.show(block=True)


if __name__ == "__main__":
    main()
