import incense
import incense.artifact
import jax
import nlopt
import numpy as np
from dotenv import load_dotenv
from incense import ExperimentLoader
from incense.artifact import PickleArtifact as PA
from matplotlib import pyplot as plt
from sacred import Experiment

from experiments.utils import *
from metatop import V_DICT
from metatop.fem_profiler import fem_profiler
from metatop.filters import setup_filter
from metatop.Metamaterial import setup_metamaterial
from metatop.optimization import OptimizationState
from metatop.optimization.epigraph import (EigenvalueProblemConstraints,
                                           EigenvectorConstraint,
                                           EpigraphOptimizer,
                                           ExtremalConstraints,
                                           TraceConstraint)
from metatop.profiling import ProfileConfig

jax.config.update("jax_enable_x64", True)


np.set_printoptions(precision=5)

# use if we want to connect to the AWS db
# load_dotenv()

ex = Experiment('eigvalprob')


@ex.config
def config():
    E_max, E_min, nu = 1., 1/30., 0.4
    start_beta, n_betas = 8, 4
    n_epochs, epoch_duration, starting_epoch_duration = 4, 50, None
    starting_epoch_duration = starting_epoch_duration or 2*epoch_duration
    extremal_mode = 1
    basis_v = 'BULK'
    objective_type = 'ray'
    nelx = nely = 50
    norm_filter_radius = 0.1
    verbose = False
    interim_plot = True
    vector_constraint = True
    tighten_vector_constraint = True
    g_vec_eps = 1.
    trace_constraint = False
    g_trc_bnd = 0.3
    weight_scaling_factor = 1.
    init_run_idx = None  # if we want to start the run with the final output density of a previous run, this is the index in the mongodb that we want to grab the output density from
    single_sim = False  # This is if we want to just run a single sim at a given param set, and not run the full optimization. We do this because we want to track the results in the database and it is easier than setting a bunch of parameters outside the experiment and calling them there. It only changes things so that the two for loops that execute the optimization run once and that nlopt actually only runs once as well. All other parameters still need to be set by the user
    enable_profiling = False
    log_to_db = True


@ex.config_hook
def connect_mongodb(config, command_name, logger):
    if config['log_to_db']:
        ex.observers.append(setup_mongo_observer(MONGO_URI, MONGO_DB_NAME))
    else:
        print("MongoDB logging is disabled for this run")
    return config


@ex.automain
def main(E_max, E_min, nu, start_beta, n_betas, n_epochs, epoch_duration, starting_epoch_duration, extremal_mode, basis_v, objective_type, nelx, nely, norm_filter_radius, verbose, interim_plot, vector_constraint, tighten_vector_constraint, g_vec_eps, trace_constraint, g_trc_bnd, weight_scaling_factor, init_run_idx, single_sim, enable_profiling, seed):

    ProfileConfig.enabled = enable_profiling
    if single_sim:
        print("SINGLE IS ENABLED. NOT RUNNING OPTIMIZATION. SETTINGS ARE FOR ONLY A SINGLE FORWARD SIMULATION")
        n_betas = 1
        n_epochs = 1
        epoch_duration = 1
        starting_epoch_duration = epoch_duration
        interim_plot = False
        vector_constraint = False
        tighten_vector_constraint = False
        g_vec_eps = 1.
        trace_constraint = False
        g_trc_bnd = 1.
        weight_scaling_factor = 1.

    run_id, outname = generate_output_dir(
        ex, extremal_mode, basis_v, seed)

    if extremal_mode == 1:
        weights = np.array([weight_scaling_factor, 1., 1.])
    else:
        weights = np.array([1., weight_scaling_factor, weight_scaling_factor])

    betas = [start_beta * 2 ** i for i in range(n_betas)]
    # ===== Component Setup =====
    metamate = setup_metamaterial(E_max,
                                  E_min,
                                  nu,
                                  nelx,
                                  nely,
                                  mesh_cell_type='tri',
                                  domain_shape='square')
    metamate.enable_profiling = ProfileConfig.enabled
    img_rez = (200, 200)
    img_shape = (metamate.width, metamate.height)

    filt, filt_fn = setup_filter(metamate, norm_filter_radius)

    # global optimization state
    ops = OptimizationState(beta=start_beta,
                            eta=0.5,
                            filt=filt,
                            filt_fn=filt_fn,
                            epoch_iter_tracker=[1])

    x = np.random.uniform(0., 1., size=metamate.R.dim())
    basis_v = V_DICT[basis_v]
    x = np.concatenate([x, basis_v.flatten(), np.ones(1)])
    # ===== End Component Setup =====

    # ===== Optimizer setup ======
    g_eig = EigenvalueProblemConstraints(basis_v,
                                         ops,
                                         metamate,
                                         extremal_mode,
                                         weights=weights,
                                         check_valid=True,
                                         plot_interval=max(
                                             epoch_duration//2, 1),
                                         show_plot=interim_plot,
                                         verbose=verbose,
                                         eps=g_vec_eps
                                         )

    opt = EpigraphOptimizer(nlopt.LD_MMA, x.size)
    opt.active_constraints = [g_eig, ]
    # opt.active_constraints.append(g_vec) if vector_constraint else None
    # opt.active_constraints.append(g_trc) if trace_constraint else None
    opt.setup()
    opt.set_maxeval(starting_epoch_duration)
    opt.set_lower_bounds(np.hstack([np.zeros(x.size-10),
                                    -np.ones(9),
                                    [-np.inf]]))
    opt.set_upper_bounds(np.hstack([np.ones(x.size-10),
                                    np.ones(9),
                                    [np.inf]]))
    # ===== End Optimizer setup ======

    # ===== Optimization Loop =====
    x_history = [x.copy()]
    for i in range(n_epochs):
        for n, beta in enumerate(betas, 1):
            run_optimization(epoch_duration, betas, ops, x,
                             g_eig, opt, x_history, n, beta)
            fem_profiler.report()

        print_epoch_summary(opt, i)
        log_and_save_results(ex, run_id, outname, metamate,
                             img_rez, img_shape, ops, x, g_eig, i)

        g_eig.eps = g_eig.eps / 10.
        # g_vec.eps = g_vec.eps / 10 if tighten_vector_constraint else g_vec.eps

    # ===== End Optimization Loop =====

    # ===== Post-Processing =====
    save_final_results(ex,
                       run_id,
                       outname,
                       metamate,
                       img_rez,
                       img_shape,
                       ops,
                       x,
                       g_eig,
                       x_history)

    if g_eig.show_plot:
        plt.close(g_eig.fig)
