
import jax

jax.config.update("jax_enable_x64", True)
import nlopt
import numpy as np
from matplotlib import pyplot as plt

from metatop import V_DICT
from metatop.filters import jax_projection, setup_filter
from metatop.helpers import init_density, update_t
from metatop.image import bitmapify
from metatop.mechanics import anisotropy_index, calculate_elastic_constants
from metatop.metamaterial import setup_metamaterial
from metatop.optimization import OptimizationState
from metatop.optimization.epigraph import (EigenvectorConstraint, Epigraph,
                                           EpigraphOptimizer,
                                           ExtremalConstraints)

np.set_printoptions(precision=5)

def main():
    E_max, E_min, nu = 1., 1e-2, 0.45
    start_beta, n_betas = 8, 4
    epoch_duration = 50
    extremal_mode = 1
    basis_v = 'BULK'
    nelx = nely = 50
    norm_filter_radius = 0.1

    # np.random.seed(73056963)
    np.random.seed(1)

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
    f = Epigraph()
    g_ext = ExtremalConstraints(v=v,
                                extremal_mode=extremal_mode,
                                metamaterial=metamate,
                                ops=ops,
                                plot_interval=10)
    g_vec = EigenvectorConstraint(v=v, ops=ops, eps=1e-1, verbose=True)

    opt = EpigraphOptimizer(nlopt.LD_MMA, x.size)
    opt.active_constraints = [g_ext, g_vec]
    opt.setup()
    opt.set_maxeval(2*epoch_duration)
    # ===== End Optimizer setup ======

    # ===== Optimization Loop =====
    for n, beta in enumerate(betas, 1):
        ops.beta, ops.epoch = beta, n
        x[:] = opt.optimize(x)
        ops.epoch_iter_tracker.append(len(g_ext.evals))

        g_vec.eps /= 10.

        print(f"\n===== Epoch Summary: {n} =====")
        print(f"Final Objective: {opt.last_optimum_value():.3f}")
        print(f"Result Code: {opt.last_optimize_result()}")
        print(f"===== End Epoch Summary: {n} =====\n")

        if n == 1:
            opt.set_maxeval(epoch_duration)
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
    print('Final ASU:', ASU)
    print('Final Elastic Constants:', elastic_constants)
    
    img_rez = 200
    img_shape = (metamate.width, metamate.height)
    x_img = 1 - np.flip(bitmapify(metamate.x,
                              img_shape,
                              (img_rez, img_rez),),
                    axis=0)
    fname = 'epigraph'
    fname += f'_v_{basis_v}'
    fname += f'_ext_{extremal_mode}'
    plt.imsave(f"output/{fname}_testbed.png", x_img, cmap='gray')
    plt.imsave(f"output/{fname}_array_testbed.png", np.tile(x_img, (4,4)), cmap='gray')
    plt.show(block=True)

if __name__ == '__main__':
    main()