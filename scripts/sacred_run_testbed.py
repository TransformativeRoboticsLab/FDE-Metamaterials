import os
import pickle

import jax

jax.config.update("jax_enable_x64", True)
import nlopt
import numpy as np
from matplotlib import pyplot as plt

from metatop import V_DICT
from metatop.filters import jax_projection, setup_filter
from metatop.image import bitmapify
from metatop.mechanics import (anisotropy_index, calculate_elastic_constants,
                               matrix_invariants)
from metatop.metamaterial import setup_metamaterial
from metatop.optimization import OptimizationState
from metatop.optimization.epigraph import (EigenvectorConstraint,
                                           EpigraphOptimizer,
                                           ExtremalConstraints)

np.set_printoptions(precision=5)

def main():
    E_max, E_min, nu = 1., 1./60., 0.45
    start_beta, n_betas = 8, 4
    n_epochs, epoch_duration = 4, 50
    extremal_mode = 1
    basis_v = 'BULK'
    objective_type = 'norm' # rayleigh or norm or ratio
    nelx = nely = 50
    norm_filter_radius = 0.1
    verbose = interim_plot = True
    seed = 1 # 916723353 # 689993214
    weights = np.array([.5, 1., 1.])
    vector_constraint = True
    tighten_vector_constraint = True

    np.random.seed(seed)

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
                                plot=interim_plot,
                                verbose=verbose,
                                w=weights,
                                objective_type=objective_type)
    g_vec = EigenvectorConstraint(v=v, 
                                  ops=ops, 
                                  eps=1., 
                                  verbose=verbose)

    opt = EpigraphOptimizer(nlopt.LD_MMA, x.size)
    opt.active_constraints = [g_ext, ]
    opt.active_constraints.append(g_vec) if vector_constraint else None
    opt.setup()
    opt.set_maxeval(2*epoch_duration)
    # ===== End Optimizer setup ======

    # ===== Optimization Loop =====
    x_history = [x.copy()]
    for _ in range(n_epochs):
        for n, beta in enumerate(betas, 1):
            ops.beta, ops.epoch = beta, n
            x[:] = opt.optimize(x)
            x_history.append(x.copy())
            ops.epoch_iter_tracker.append(len(g_ext.evals))
            opt.set_maxeval(epoch_duration)

        print(f"\n===== Epoch Summary: {n} =====")
        print(f"Final Objective: {opt.last_optimum_value():.3f}")
        print(f"Result Code: {opt.last_optimize_result()}")
        print(f"===== End Epoch Summary: {n} =====\n")
        
        g_vec.eps = g_vec.eps / 10 if tighten_vector_constraint else g_vec.eps

    g_ext.update_plot(x[:-1])

    # ===== End Optimization Loop =====

    # ===== Post-Processing =====
    metamate.x.vector()[:] = jax_projection(filt_fn(x[:-1]), ops.beta, ops.eta)

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
    
    dirname = './output/epigraph'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fname = f'{basis_v}'
    fname += f'_m_{extremal_mode}'
    fname += f'_seed_{seed}'
    outname = dirname + '/' + fname

    with open(f'{outname}_testbed.pkl', 'wb') as f:
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
    plt.imsave(f"{outname}_testbed.png", x_img, cmap='gray')
    plt.imsave(f"{outname}_array_testbed.png", np.tile(x_img, (4,4)), cmap='gray')
    plt.show(block=True)

if __name__ == '__main__':
    main()