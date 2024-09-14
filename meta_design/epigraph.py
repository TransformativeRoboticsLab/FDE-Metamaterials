from functools import partial

import jax
import nlopt
import numpy as np
from fenics import *
from filters import (DensityFilter, HelmholtzFilter, jax_density_filter,
                     jax_helmholtz_filter, jax_projection)
from helpers import init_density
from image import bitmapify
from matplotlib import pyplot as plt
from metamaterial import Metamaterial
from optimization import (EigenvectorConstraint, Epigraph, ExtremalConstraints,
                          InvariantsConstraint, OptimizationState)

from mechanics import anisotropy_index, calculate_elastic_constants

jax.config.update("jax_enable_x64", True)

np.set_printoptions(precision=5)
# np.set_printoptions(suppress=True)

RAND_SEED = 1
print(f"Random Seed: {RAND_SEED}")
np.random.seed(RAND_SEED)

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
        new_t = max(new_t, *(results))
    x[-1] = new_t
    print(f"New t value: {x[-1]:.3e}")


def setup_metamaterial(E_max, E_min, nu, nelx, nely, mesh_cell_type='triangle', domain_shape='square'):
    metamaterial = Metamaterial(E_max, E_min, nu, nelx, nely, domain_shape=domain_shape)
    if 'tri' in mesh_cell_type:
        # metamaterial.mesh = UnitSquareMesh(nelx, nely, 'crossed')
        P0 = Point(0, 0)
        P1 = Point(1, 1)
        if 'rect' in domain_shape:
            P1 = Point(np.sqrt(3), 1)
            nelx = int(nelx * np.sqrt(3))
            print(f"Rectangular domain requested. Adjusting nelx to {nelx:d} cells to better match aspect ratio.")
        metamaterial.mesh = RectangleMesh(P0, P1, nelx, nely, 'crossed')
        metamaterial.domain_shape = domain_shape
    elif 'quad' in mesh_cell_type:
        metamaterial.mesh = RectangleMesh.create([Point(0, 0), Point(1, 1)],
                                                 [nelx, nely],
                                                 CellType.Type.quadrilateral)
    else:
        raise ValueError(f"Invalid cell_type: {mesh_cell_type}")
    metamaterial.create_function_spaces()
    return metamaterial


def main():
    # ===== Preamble =====
    # inputs
    E_max, E_min, nu = 1., 1e-2, 0.45
    vol_frac = 0.1
    start_beta, n_betas = 8, 4
    betas = [start_beta * 2 ** i for i in range(n_betas)]
    # betas.append(betas[-1]) # repeat the last beta for final epoch when we turn on constraints
    print(f"Betas: {betas}")
    eta = 0.5
    epoch_duration = 400
    basis_v = 'BULK'
    symmetry_order = 'rectangular'
    density_seed_type = 'uniform'
    extremal_mode = 1
    mesh_cell_type = 'tri'  # triangle, quadrilateral
    domain_shape = 'square' # square or rect
    if 'tri' in mesh_cell_type:
        nelx = 50
    elif 'quad' in mesh_cell_type:
        nelx = 100
    else:
        raise ValueError(f"Invalid mesh_cell_type: {mesh_cell_type}")
    nely = nelx

    cell_min_dimension = 25.
    line_width_mm = 2.5
    line_space_mm = line_width_mm
    norm_line_width = line_width_mm / cell_min_dimension
    norm_line_space = line_space_mm / cell_min_dimension
    norm_filter_radius = norm_line_width

    print(f"Cell Minimum Dimension: {cell_min_dimension} mm")
    print(f"Line Width: {line_width_mm} mm")
    print(f"Line Space: {line_space_mm} mm")
    print(f"Normalized Line Space: {norm_line_space}")
    print(f"Normalized Line Width: {norm_line_width}")
    print(f"Final Filter Radius: {norm_filter_radius}")
    # ===== End Preamble =====

    # ===== Component Setup =====
    metamate = setup_metamaterial(E_max,
                                  E_min,
                                  nu,
                                  nelx,
                                  nely,
                                  mesh_cell_type=mesh_cell_type,
                                  domain_shape=domain_shape)
    # metamate.plot_mesh(labels=True)


    # density filter setup
    if metamate.R.ufl_element().degree() > 0:
        print("Using Helmholtz filter")
        filt = HelmholtzFilter(radius=norm_filter_radius, 
                                fn_space=metamate.R)
        filter_fn = partial(jax_helmholtz_filter, filt)
    elif metamate.R.ufl_element().degree() == 0:
        print("Using Density filter")
        filt = DensityFilter(mesh=metamate.mesh,
                            radius=norm_filter_radius,
                            distance_method='periodic')
        filter_fn = partial(jax_density_filter, filt.H_jax, filt.Hs_jax)
    else:
        raise ValueError("Invalid filter type. Must be DensityFilter or HelmholtzFilter")

    # global optimization state
    ops = OptimizationState(beta=start_beta,
                            eta=eta,
                            filt=filt,
                            filt_fn = filter_fn,
                            epoch_iter_tracker=[1])

    # seeding the initial density
    x = init_density(density_seed_type, vol_frac, metamate.R.dim())
    # x, metamate.mirror_map = mirror_density(x, metamate.R, type='x')
    x = np.append(x, 1.)
    # ===== End Component Setup =====
    
    # ===== Objective and Constraints Setup =====
    v = v_dict[basis_v]
    f = Epigraph()
    g_ext = ExtremalConstraints(v=v,
                                extremal_mode=extremal_mode,
                                metamaterial=metamate,
                                ops=ops,
                                plot_interval=10)
    # g_sym = MaterialSymmetryConstraints(ops=ops, eps=1e-1, verbose=True, symmetry_order=symmetry_order)
    g_inv = InvariantsConstraint(ops=ops, verbose=True)
    g_vec = EigenvectorConstraint(v=v, ops=ops, eps=1., verbose=True)

    active_constraints = [g_ext, g_vec]
    # ===== End Objective and Constraints Setup =====

    # ===== Optimizer setup ======
    opt = nlopt.opt(nlopt.LD_MMA, x.size)

    opt.set_min_objective(f)
    for g in active_constraints:
        opt.add_inequality_mconstraint(g, np.zeros(g.n_constraints))
    # opt.add_inequality_mconstraint(g_inv, np.zeros(g_inv.n_constraints))

    opt.set_lower_bounds(np.append(np.zeros(x.size - 1), -np.inf))
    opt.set_upper_bounds(np.append(np.ones(x.size - 1), np.inf))
    opt.set_maxeval(2*epoch_duration)
    opt.set_param('dual_ftol_rel', 1e-6)
    # ===== End Optimizer setup ======

    # ===== Optimization Loop =====
    for n, beta in enumerate(betas, 1):
        ops.beta, ops.epoch = beta, n
        update_t(x, active_constraints)
        x[:] = opt.optimize(x)
        # x[:-1] = jax_projection(filter_fn(x[:-1]), ops.beta, 0.35)
        # x[:-1] += np.random.uniform(-0.2, 0.2)
        # x[:-1] = x[:-1].clip(0., 1.)

        ops.epoch_iter_tracker.append(len(g_ext.evals))
        
        # g_sym.eps /= 2.
        g_vec.eps /= 2.

        opt.set_maxeval(epoch_duration)
        

        print(f"\n===== Epoch Summary: {n} =====")
        print(f"Final Objective: {opt.last_optimum_value():.3f}")
        print(f"Result Code: {opt.last_optimize_result()}")
        print(f"===== End Epoch Summary: {n} =====\n")

    # ===== End Optimization Loop =====

    # ===== Post-Processing =====
    x = x[:-1]
    x = filter_fn(x)
    x = jax_projection(x, ops.beta, ops.eta)
    metamate.x.vector()[:] = x

    m = np.diag(np.array([1, 1, np.sqrt(2)]))
    final_C = m @ np.asarray(metamate.solve()[1]) @ m
    print('Final C:\n', final_C)
    w, v = np.linalg.eigh(final_C)
    print('Final Eigenvalues:\n', w)
    print('Final Eigenvalue Ratios:\n', w / np.max(w))
    print('Final Eigenvectors:\n', v)

    print('Final ASU:', anisotropy_index(final_C, input_style='standard')[-1])
    print('Final Elastic Constants:', calculate_elastic_constants(
        final_C, input_style='standard'))
    
    img_rez = 200
    img_shape = (metamate.width, metamate.height)
    x_img = np.flip(bitmapify(metamate.x,
                              img_shape,
                              (img_rez, img_rez),),
                    axis=0)
    plt.figure()
    plt.imshow(x_img, cmap='gray')
    fname = 'epigraph'
    fname += f'_v_{basis_v}'
    fname += f'_sym_{symmetry_order}'
    fname += f'_dens_{density_seed_type}'
    fname += f'_vf_{vol_frac:.2f}'
    fname += f'_b_{start_beta}_{n_betas}'
    fname += f'_ext_{extremal_mode}'
    fname += f'_R_{norm_filter_radius}'
    plt.imsave(f"output/{fname}.png", x_img, cmap='gray')
    plt.imsave(f"output/{fname}_array.png", np.tile(x_img, (4,4)), cmap='gray')
    plt.show(block=True)


if __name__ == "__main__":
    main()
