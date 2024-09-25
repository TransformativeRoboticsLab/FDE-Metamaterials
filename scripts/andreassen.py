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
from metatop.optimization.examples import AndreassenOptimization
from metatop.optimization.scalar import (BulkModulusConstraint,
                                         IsotropicConstraint, VolumeConstraint)

np.set_printoptions(precision=5)

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('andreassen')
ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='andreassen'))

@ex.config
def config():
    E_max, E_min, nu = 1., 1e-9, 0.3
    start_beta, n_betas = 2, 7
    penalty = 3.
    epoch_duration = 100
    nelx = nely = 100
    norm_filter_radius = 0.1
    verbose = interim_plot = True
    obj_type = 'pr'
    vol_frac = 0.35
    bulk_modulus_ratio = 0.2*1e-2 # 0.2% of base K
    iso_eps = 1e-5

@ex.automain
def main(E_max, E_min, nu, start_beta, n_betas, penalty, epoch_duration, nelx, nely, norm_filter_radius, verbose, interim_plot, obj_type, vol_frac, bulk_modulus_ratio, iso_eps, seed):
    
    dirname = './output/andreassen'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fname = f'{ex.current_run._id}'
    fname += f'_{obj_type}'
    fname += f'_seed_{seed}'
    outname = dirname + '/' + fname
    
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
    img_rez = (nelx, nely)
    img_shape = (metamate.width, metamate.height)

    filt, filt_fn = setup_filter(metamate, norm_filter_radius)

    # global optimization state
    ops = OptimizationState(beta=1,
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
                               plot=interim_plot,
                               plot_interval=epoch_duration//2)
    
    g_vol = VolumeConstraint(vol_frac, ops=ops, verbose=verbose)
    g_bulk = BulkModulusConstraint(E_max, nu, bulk_modulus_ratio, ops=ops, verbose=verbose)
    g_iso = IsotropicConstraint(iso_eps, ops, verbose=verbose)

    opt = nlopt.opt(nlopt.LD_MMA, x.size)
    opt.set_min_objective(f)
    opt.add_inequality_constraint(g_vol, 0.)
    opt.add_inequality_constraint(g_bulk, 0.)
    opt.add_inequality_constraint(g_iso, 0.)
    
    opt.set_lower_bounds(np.zeros(x.size))
    opt.set_upper_bounds(np.ones(x.size))
    # opt.set_param('dual_ftol_rel', 1e-6)
    # ===== End Optimization Setup =====

    # ===== Optimization Loop =====
    # First epoch lets optimizer converge at beta=1
    # Previous runs indicate this takes about 4k iterations
    x_history = [x.copy()]
    opt.set_maxeval(4_000)
    x[:] = opt.optimize(x)
    x_history.append(x.copy())
    ops.epoch_iter_tracker.append(len(f.evals))

    log_values(ex, forward_solve(x, metamate, ops, simp=True))
    x_img = bitmapify(metamate.x, img_shape, img_rez, invert=True)
    fcellname = f"{outname}_cell_e-1.png"
    plt.imsave(fcellname, x_img, cmap='gray')
    ex.add_artifact(fcellname)
    ftimelinename = f"{outname}_timeline_e-1.png"
    f.fig.savefig(ftimelinename)
    ex.add_artifact(ftimelinename)

    # Now we can start the epochs
    opt.set_maxeval(epoch_duration)
    for n, beta in enumerate(betas, 2):
        ops.beta, ops.epoch = beta, n
        x[:] = opt.optimize(x)
        x_history.append(x.copy())
        ops.epoch_iter_tracker.append(len(f.evals))
        
        print(f"\n===== Epoch Summary: {n} =====")
        print(f"Final Objective: {opt.last_optimum_value():.3f}")
        print(f"Result Code: {opt.last_optimize_result()}")
        print(f"===== End Epoch Summary: {n} =====\n")

        metamate.x.vector()[:] = x
        log_values(ex, forward_solve(x, metamate, ops, simp=True))
        x_img = bitmapify(metamate.x, img_shape, img_rez, invert=True)
        fcellname = f"{outname}_cell_e-{n}.png"
        plt.imsave(fcellname, x_img, cmap='gray')
        ex.add_artifact(fcellname)
        ftimelinename = f"{outname}_timeline_e-{n}.png"
        f.fig.savefig(ftimelinename)
        ex.add_artifact(ftimelinename)
    # ===== End Optimization Loop =====

    # ===== Post-Processing =====
    final_C = forward_solve(x, metamate, ops, simp=True)

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
        
    x_img = bitmapify(metamate.x, img_shape, img_rez, invert=True)
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
    ex.add_artifact(f"{outname}.png")
    ex.add_artifact(f"{outname}_array.png")