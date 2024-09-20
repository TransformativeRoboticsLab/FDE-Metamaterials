import os
import pickle

import jax

jax.config.update("jax_enable_x64", True)
import nlopt
import numpy as np
from matplotlib import pyplot as plt

from metatop import V_DICT
from metatop.filters import jax_projection, setup_filter
from metatop.mechanics import (anisotropy_index, calculate_elastic_constants,
                               matrix_invariants)
from metatop.metamaterial import setup_metamaterial
from metatop.optimization import OptimizationState
from metatop.optimization.examples import AndreassenOptimization
from metatop.optimization.scalar import VolumeConstraint

np.set_printoptions(precision=5)

def main():
    E_max, E_min, nu = 1., 1e-9, 0.3
    start_beta, n_betas = 1, 8
    penalty = 3.
    epoch_duration = 50
    nelx = nely = 100
    norm_filter_radius = 0.1
    verbose = interim_plot = True
    seed = 1 # 916723353 # 689993214
    obj_type = 'pr'
    vol_frac = 0.35

    np.random.seed(seed)
    betas = [start_beta * 2 ** i for i in range(n_betas)]
    # ===== End Preamble =====

    # ===== Component Setup =====
    metamate = setup_metamaterial(E_max,
                                  E_min,
                                  nu,
                                  nelx,
                                  nely,
                                  mesh_cell_type='quad',
                                  domain_shape='square')

    filt, filt_fn = setup_filter(metamate, norm_filter_radius)

    # global optimization state
    ops = OptimizationState(beta=start_beta,
                            eta=0.5,
                            filt=filt,
                            filt_fn = filt_fn,
                            epoch_iter_tracker=[1],
                            pen=penalty)

    # seeding the initial density
    x = np.random.uniform(0., 1., metamate.R.dim()) * (vol_frac * 2)
    # ===== End Component Setup =====
    
    # ===== Optimizer setup ======
    f = AndreassenOptimization(obj_type=obj_type,
                               metamaterial=metamate,
                               ops=ops,
                               verbose=verbose,
                               plot=interim_plot,)
    
    g_vol = VolumeConstraint(vol_frac, ops=ops, verbose=verbose)

    opt = nlopt.opt(nlopt.LD_MMA, x.size)
    opt.set_min_objective(f)
    opt.add_inequality_constraint(g_vol, 0.)
    
    opt.set_lower_bounds(np.zeros(x.size))
    opt.set_upper_bounds(np.ones(x.size))
    opt.set_maxeval(epoch_duration)
    opt.set_param('dual_ftol_rel', 1e-6)
    # ===== End Optimization Setup =====

    # ===== Optimization Loop =====
    for n, beta in enumerate(betas, 1):
        ops.beta, ops.epoch = beta, n
        x[:] = opt.optimize(x)
        ops.epoch_iter_tracker.append(len(f.evals))
        
        print(f"\n===== Epoch Summary: {n} =====")
        print(f"Final Objective: {opt.last_optimum_value():.3f}")
        print(f"Result Code: {opt.last_optimize_result()}")
        print(f"===== End Epoch Summary: {n} =====\n")

        opt.set_maxeval(epoch_duration)
    # ===== End Optimization Loop =====

    # ===== Post-Processing =====
    metamate.x.vector()[:] = jax_projection(filt_fn(x), ops.beta, ops.eta)

    m = np.diag(np.array([1, 1, np.sqrt(2)]))
    final_C = m @ np.asarray(metamate.solve()[1]) @ m
    w, v = np.linalg.eigh(final_C)
    print('Final C:\n', final_C)
    print('Final Eigenvalues:\n', w)
    print('Final Eigenvalue Ratios:\n', w / np.max(w))
    print('Final Eigenvectors:\n', v)

    ASU = anisotropy_index(final_C, input_style='mandel')
    elastic_constants = calculate_elastic_constants(final_C, input_style='mandel')
    invariants = matrix_invariants(final_C)
    print('Final ASU:', ASU)
    print('Final Elastic Constants:', elastic_constants)
    print('Final Invariants:', invariants)
    plt.show(block=True)

if __name__ == '__main__':
    main()