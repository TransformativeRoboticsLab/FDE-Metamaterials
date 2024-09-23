import os
import pickle

import jax

jax.config.update("jax_enable_x64", True)
import nlopt
import numpy as np
from matplotlib import pyplot as plt

from metatop import V_DICT
from metatop.filters import jax_projection, setup_filter
from metatop.helpers import forward_solve, log_values
from metatop.image import bitmapify
from metatop.mechanics import (anisotropy_index, calculate_elastic_constants,
                               matrix_invariants)
from metatop.metamaterial import setup_metamaterial
from metatop.optimization import OptimizationState
from metatop.optimization.epigraph import (EigenvectorConstraint,
                                           EpigraphOptimizer,
                                           ExtremalConstraints)

np.set_printoptions(precision=5)

from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver

ex = Experiment('metatop_epigraph')
ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='metatop'))

@ex.config
def config():
    E_max, E_min, nu = 1., 1./60., 0.45
    start_beta, n_betas = 8, 4
    n_epochs, epoch_duration = 4, 50
    extremal_mode = 1
    basis_v = 'BULK'
    objective_type = 'norm' # rayleigh or norm or ratio
    nelx = nely = 50
    norm_filter_radius = 0.1
    verbose = interim_plot = True
    vector_constraint = True
    tighten_vector_constraint = True
    weight_scaling_factor = 1

@ex.automain
def main(E_max, E_min, nu, start_beta, n_betas, n_epochs, epoch_duration, extremal_mode, basis_v, objective_type, nelx, nely, norm_filter_radius, verbose, interim_plot, vector_constraint, tighten_vector_constraint, g_vec_eps, seed):

    dirname = './output/epigraph'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fname = f'{basis_v}'
    fname += f'_m_{extremal_mode}'
    fname += f'_seed_{seed}'
    outname = dirname + '/' + fname

    weights = np.array([weight_scaling_factor, 1., 1.]) if extremal_mode == 1 else np.array([1., weight_scaling_factor, weight_scaling_factor])
    betas = [start_beta * 2 ** i for i in range(n_betas)]
    # ===== Component Setup =====
    metamate = setup_metamaterial(E_max,
                                  E_min,
                                  nu,
                                  nelx,
                                  nely,
                                  mesh_cell_type='tri',
                                  domain_shape='square')

    filt, filt_fn = setup_filter(metamate, norm_filter_radius)

    # global optimization state
    ops = OptimizationState(beta=start_beta,
                            eta=0.5,
                            filt=filt,
                            filt_fn = filt_fn,
                            epoch_iter_tracker=[1])

    # seeding the initial density
    x = np.random.uniform(0., 1., metamate.R.dim())
    x = np.append(x, 1.)
    # ===== End Component Setup =====
    
    # ===== Optimizer setup ======
    v = V_DICT[basis_v]
    g_ext = ExtremalConstraints(v=v,
                                extremal_mode=extremal_mode,
                                metamaterial=metamate,
                                ops=ops,
                                plot_interval=10,
                                show_plot=interim_plot,
                                verbose=verbose,
                                w=weights,
                                objective_type=objective_type)
    g_vec = EigenvectorConstraint(v=v, 
                                  ops=ops, 
                                  eps=g_vec_eps, 
                                  verbose=verbose)

    opt = EpigraphOptimizer(nlopt.LD_MMA, x.size)
    opt.active_constraints = [g_ext, ]
    opt.active_constraints.append(g_vec) if vector_constraint else None
    opt.setup()
    opt.set_maxeval(2*epoch_duration)
    # ===== End Optimizer setup ======

    # ===== Optimization Loop =====
    x_history = [x.copy()]
    for i in range(n_epochs):
        for n, beta in enumerate(betas, 1):
            ops.beta, ops.epoch = beta, n
            x[:] = opt.optimize(x)
            x_history.append(x.copy())
            opt.set_maxeval(epoch_duration)

        ops.epoch_iter_tracker.append(len(g_ext.evals))
        print(f"\n===== Epoch Summary: {n} =====")
        print(f"Final Objective: {opt.last_optimum_value():.3f}")
        print(f"Result Code: {opt.last_optimize_result()}")
        print(f"===== End Epoch Summary: {n} =====\n")
        
        g_vec.eps = g_vec.eps / 10 if tighten_vector_constraint else g_vec.eps

        g_ext.update_plot(x[:-1])
        g_ext.fig.savefig(f"{outname}_timeline_e-{i+1}.png")
        ex.add_artifact(f"{outname}_timeline_e-{i+1}.png")
        log_values(ex, forward_solve(x[:-1], metamate, ops))

    # ===== End Optimization Loop =====

    # ===== Post-Processing =====
    final_C = forward_solve(x[:-1], metamate, ops)
    
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

    with open(f'{outname}.pkl', 'wb') as f:
        pickle.dump({'x': x,
                     'x_history': x_history},
                    f)

    img_rez = 200
    img_shape = (metamate.width, metamate.height)
    x_img = 1 - np.flip(bitmapify(metamate.x,
                              img_shape,
                              (img_rez, img_rez),),
                    axis=0)
    g_ext.fig.savefig(f"{outname}_timeline.png")
    plt.imsave(f"{outname}.png", x_img, cmap='gray')
    plt.imsave(f"{outname}_array.png", np.tile(x_img, (4,4)), cmap='gray')

    ex.info['final_C'] = final_C
    ex.info['eigvals'] = w
    ex.info['norm_eigvals'] = w / np.max(w)
    ex.info['eigvecs'] = v
    ex.info['ASU'] = ASU
    ex.info['elastic_constants'] = elastic_constants
    ex.info['invariants'] = invariants
    ex.add_artifact(f'{outname}.pkl')
    ex.add_artifact(f'{outname}_timeline.png')
    ex.add_artifact(f"output/epigraph/{fname}.png")
    ex.add_artifact(f"output/epigraph/{fname}_array.png")