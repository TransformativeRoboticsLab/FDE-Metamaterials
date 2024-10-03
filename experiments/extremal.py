import jax

jax.config.update("jax_enable_x64", True)

import nlopt
import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sacred import Experiment

from experiments.utils import *
from metatop import V_DICT
from metatop.filters import setup_filter
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

    run_id, outname = generate_output_filepath(ex, extremal_mode, basis_v, seed)

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

        print_epoch_summary(opt, i)
        log_and_save_results(ex, run_id, outname, metamate, img_rez, img_shape, ops, x, g_ext, i)

        g_vec.eps = g_vec.eps / 10 if tighten_vector_constraint else g_vec.eps

    # ===== End Optimization Loop =====

    # ===== Post-Processing =====
    save_results(ex, 
                 run_id, 
                 outname, 
                 metamate, 
                 img_rez, 
                 img_shape, 
                 ops, 
                 x, 
                 g_ext, 
                 x_history)

    if g_ext.show_plot:
        plt.close(g_ext.fig)
