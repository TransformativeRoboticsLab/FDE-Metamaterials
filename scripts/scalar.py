from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
import nlopt
import numpy as np
from matplotlib import pyplot as plt

from metatop import V_DICT
from metatop.filters import (DensityFilter, HelmholtzFilter,
                             jax_density_filter, jax_helmholtz_filter,
                             jax_projection)
from metatop.helpers import init_density
from metatop.mechanics import anisotropy_index, calculate_elastic_constants
from metatop.metamaterial import setup_metamaterial
from metatop.optimization import (EnergyConstraints, EnergyObjective,
                                  OptimizationState)

np.set_printoptions(precision=5)
# np.set_printoptions(suppress=True)

RAND_SEED = 1
print(f"Random Seed: {RAND_SEED}")
np.random.seed(RAND_SEED)

def main():
    E_max, E_min, nu = 1., 1e-9, 0.45
    vol_frac = 0.1
    start_beta, n_betas = 8, 4
    betas = [start_beta * 2 ** i for i in range(n_betas)]
    # betas.append(betas[-1]) # repeat the last beta for final epoch when we turn on constraints
    print(f"Betas: {betas}")
    eta = 0.5
    epoch_duration = 50
    basis_v = 'VERT'
    density_seed_type = 'uniform'
    extremal_mode = 1
    mesh_cell_type = 'tri'  # triangle, quadrilateral
    if 'tri' in mesh_cell_type:
        nelx = 50
    elif 'quad' in mesh_cell_type:
        nelx = 100
        # nelx = 50
    else:
        raise ValueError(f"Invalid mesh_cell_type: {mesh_cell_type}")
    nely = nelx

    cell_side_length_mm = 25.
    line_width_mm = 2.5
    line_space_mm = line_width_mm
    norm_line_width = line_width_mm / cell_side_length_mm
    norm_line_space = line_space_mm / cell_side_length_mm
    norm_filter_radius = norm_line_width

    print(f"Cell Side Length: {cell_side_length_mm} mm")
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
                                  mesh_cell_type=mesh_cell_type)


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
    # x = mirror_density(x, metamate.mesh)
    # ===== End Component Setup =====
    
    # ===== Objective and Constraints Setup =====
    v = V_DICT[basis_v]
    f = EnergyObjective(v=v, extremal_mode=extremal_mode, metamaterial=metamate, ops=ops)
    g = EnergyConstraints(a=2., v=v, extremal_mode=extremal_mode, ops=ops, verbose=True)
    
    # ===== End Objective and Constraints Setup =====
    
    # ===== Optimization Setup =====
    opt = nlopt.opt(nlopt.LD_MMA, x.size)
    opt.set_min_objective(f)
    opt.add_inequality_mconstraint(g, np.zeros(g.n_constraints))
    # opt.add_inequality_mconstraint(g_vec, np.zeros(g_vec.n_constraints))
    
    opt.set_lower_bounds(np.zeros(x.size))
    opt.set_upper_bounds(np.ones(x.size))
    opt.set_maxeval(epoch_duration)
    opt.set_param('dual_ftol_rel', 1e-6)
    # ===== End Optimization Setup =====

    # ===== Optimization Loop =====
    # progressively up the projection
    for n, beta in enumerate(betas, 1):
        ops.beta, ops.epoch = beta, n
        x[:] = np.copy(opt.optimize(x))
        print(f"\n===== Epoch Summary: {n} =====")
        print(f"Final Objective: {opt.last_optimum_value():.3f}")
        print(f"Result Code: {opt.last_optimize_result()}")
        print(f"===== End Epoch Summary: {n} =====\n")
        ops.epoch_iter_tracker.append(len(f.evals))
        
        g.a *= 10.

    # ===== End Optimization Loop =====
            
    # ===== Post-Optimization Analysis =====
    x = filter_fn(x)
    x = jax_projection(x, ops.beta, ops.eta)
    metamate.x.vector()[:] = x
    
    m = np.diag(np.array([1., 1., np.sqrt(2)]))
    final_C = m @ np.asarray(metamate.solve()[1]) @ m
    print('Final C:\n', final_C)
    w, v = np.linalg.eigh(final_C)
    print('Final Eigenvalues:\n', w)
    print('Final Eigenvalue Ratios:\n', w / np.max(w))
    print('Final Eigenvectors:\n', v)

    print('Final ASU:', anisotropy_index(final_C, input_style='standard')[-1])
    print('Final Elastic Constants:', calculate_elastic_constants(final_C, input_style='standard'))
        
    plt.show(block=True)


if __name__ == "__main__":
    main()