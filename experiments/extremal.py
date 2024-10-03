import jax

jax.config.update("jax_enable_x64", True)

import os
import pickle
import sys

import nlopt
import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sacred import Experiment

from experiments.utils import (forward_solve, log_values, run_optimization,
                               save_bmp_and_artifact, save_fig_and_artifact,
                               setup_mongo_observer)
from metatop import V_DICT
from metatop.filters import setup_filter
from metatop.image import bitmapify
from metatop.mechanics import (anisotropy_index, calculate_elastic_constants,
                               matrix_invariants)
from metatop.metamaterial import setup_metamaterial
from metatop.optimization import OptimizationState
from metatop.optimization.epigraph import (EigenvectorConstraint,
                                           EpigraphOptimizer,
                                           ExtremalConstraints,
                                           TraceConstraint)

np.set_printoptions(precision=5)

load_dotenv()
# mongo_uri = os.getenv('MONGO_URI')
mongo_uri = 'mongodb://localhost:27017'

ex = Experiment('extremal')
ex.observers.append(setup_mongo_observer(mongo_uri, 'metatop'))

@ex.config
def config():
    E_max, E_min, nu = 1., 1./60., 0.45
    start_beta, n_betas = 8, 4
    n_epochs, epoch_duration, starting_epoch_duration = 4, 50, None
    starting_epoch_duration = starting_epoch_duration or 2*epoch_duration
    extremal_mode = 1
    basis_v = 'BULK'
    objective_type = 'ray_sq' # rayleigh or norm or ratio
    nelx = nely = 50
    norm_filter_radius = 0.1
    verbose = False
    interim_plot = True
    vector_constraint = True
    tighten_vector_constraint = True
    g_vec_eps = 1.
    trace_constraint = True
    g_trc_bnd = 0.3
    weight_scaling_factor = 1.

@ex.automain
def main(E_max, E_min, nu, start_beta, n_betas, n_epochs, epoch_duration, starting_epoch_duration, extremal_mode, basis_v, objective_type, nelx, nely, norm_filter_radius, verbose, interim_plot, vector_constraint, tighten_vector_constraint, g_vec_eps, trace_constraint, g_trc_bnd, weight_scaling_factor, seed):

    run_id = ex.current_run._id
    dirname = './output/epigraph'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fname = str(run_id)
    fname += f'_{basis_v}'
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
    img_rez = (200, 200)
    img_shape = (metamate.width, metamate.height)

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
                                plot_interval=epoch_duration//2,
                                show_plot=interim_plot,
                                verbose=verbose,
                                w=weights,
                                objective_type=objective_type)
    g_vec = EigenvectorConstraint(v=v, 
                                  ops=ops, 
                                  eps=g_vec_eps, 
                                  verbose=verbose)
    g_trc = TraceConstraint(ops=ops, bound=g_trc_bnd, verbose=verbose)

    opt = EpigraphOptimizer(nlopt.LD_MMA, x.size)
    opt.active_constraints = [g_ext, ]
    opt.active_constraints.append(g_vec) if vector_constraint else None
    opt.active_constraints.append(g_trc) if trace_constraint else None
    opt.setup()
    opt.set_maxeval(starting_epoch_duration)
    # ===== End Optimizer setup ======

    # ===== Optimization Loop =====
    x_history = [x.copy()]
    for i in range(n_epochs):
        for n, beta in enumerate(betas, 1):
            run_optimization(epoch_duration, betas, ops, x, g_ext, opt, x_history, n, beta)

        print(f"\n===== Epoch Summary: {i+1} =====")
        print(f"Final Objective: {opt.last_optimum_value():.3f}")
        print(f"Result Code: {opt.last_optimize_result()}")
        print(f"===== End Epoch Summary: {i+1} =====\n")
        
        g_vec.eps = g_vec.eps / 10 if tighten_vector_constraint else g_vec.eps

        g_ext.update_plot(x[:-1])
        save_fig_and_artifact(ex, g_ext.fig, outname, f'{run_id}_timeline_e-{i+1}.png')

        metamate.x.vector()[:] = x[:-1]
        ex.log_scalar('volume_fraction', metamate.volume_fraction)
        log_values(ex, forward_solve(x[:-1], metamate, ops))

        x_img = bitmapify(metamate.x, img_shape, img_rez, invert=True)
        save_bmp_and_artifact(ex, x_img, outname, f'{run_id}_cell_e-{i+1}.png')

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
                     'x_history': x_history,
                     'evals': g_ext.evals},
                    f)

    save_fig_and_artifact(ex, g_ext.fig, outname, f'{run_id}_timeline.png')
    x_img = bitmapify(metamate.x, img_shape, img_rez, invert=True)
    save_bmp_and_artifact(ex, x_img, outname, f'{run_id}_cell.png')
    save_bmp_and_artifact(ex, np.tile(x_img, (4,4)), outname, f'{run_id}_array.png')

    ex.info['final_C'] = final_C
    ex.info['eigvals'] = w
    ex.info['norm_eigvals'] = w / np.max(w)
    ex.info['eigvecs'] = v
    ex.info['ASU'] = ASU
    ex.info['elastic_constants'] = elastic_constants
    ex.info['invariants'] = invariants
    ex.add_artifact(f'{outname}.pkl')
    if g_ext.show_plot:
        plt.close(g_ext.fig)