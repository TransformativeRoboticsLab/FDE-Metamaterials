import incense
import incense.artifact
import jax
import nlopt
import numpy as np
from dotenv import load_dotenv
from incense import ExperimentLoader
from incense.artifact import PickleArtifact as PA
from loguru import logger
from matplotlib import pyplot as plt
from sacred import Experiment

from experiments.utils import *
from metatop import V_DICT
from metatop.fem_profiler import fem_profiler
from metatop.filters import setup_filter
from metatop.Metamaterial import setup_metamaterial
from metatop.optimization import OptimizationState
from metatop.optimization.epigraph import (EigenvectorEpigraphConstraint,
                                           EpigraphObjective,
                                           EpigraphOptimizer,
                                           PrimaryEpigraphConstraint)
from metatop.profiling import ProfileConfig
from metatop.utils import mirror_density

jax.config.update("jax_enable_x64", True)


file_name = os.path.basename(__file__)
log_file = f"{file_name}.log"
logger.configure(
    handlers=[
        {"sink": f"{file_name}.log",
         "rotation": "500 MB",
         "level": "DEBUG",
         },
        {"sink": sys.stderr,
         "level": "DEBUG",
         "format": "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <white>{message}</white>"}
    ]
)
logger.info(f"{file_name} started")

np.set_printoptions(precision=4)

# use if we want to connect to the AWS db
# load_dotenv()

ex = Experiment('extremal')


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
    init_run_idx = None  # if we want to start the run with the final output density of a previous run, this is the index in the mongodb that we want to grab the output density from
    single_sim = False  # This is if we want to just run a single sim at a given param set, and not run the full optimization. We do this because we want to track the results in the database and it is easier than setting a bunch of parameters outside the experiment and calling them there. It only changes things so that the two for loops that execute the optimization run once and that nlopt actually only runs once as well. All other parameters still need to be set by the user
    enable_profiling = False
    log_to_db = True


@ex.config_hook
def startup(config, command_name, logger):
    # connecto to MongoDB
    if config['log_to_db']:
        ex.observers.append(setup_mongo_observer(MONGO_URI, MONGO_DB_NAME))
    else:
        print("MongoDB logging is diabled for this run")

    ProfileConfig.enabled = config['enable_profiling']

    return config


@ex.automain
def main(E_max, E_min, nu, start_beta, n_betas, n_epochs, epoch_duration, starting_epoch_duration, extremal_mode, basis_v, objective_type, nelx, nely, norm_filter_radius, verbose, interim_plot, vector_constraint, tighten_vector_constraint, g_vec_eps, trace_constraint, g_trc_bnd, init_run_idx, single_sim, enable_profiling, log_to_db, seed):

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
    ops = OptimizationState(basis_v=V_DICT[basis_v],
                            extremal_mode=extremal_mode,
                            metamaterial=metamate,
                            filt=filt,
                            filt_fn=filt_fn,
                            beta=start_beta,
                            eta=0.5,
                            img_shape=(metamate.width, metamate.height),
                            img_resolution=(200, 200),
                            plot_interval=25,
                            )

    # x = seed_density(init_run_idx, metamate.R.dim())
    x = np.random.uniform(0., 1., size=metamate.R.dim())
    x = np.append(x, 1.)  # append t for epigraph
    x_history = [x.copy()]

    # ===== End Component Setup =====

    # ===== Optimizer setup ======
    f = EpigraphObjective(ops)
    g1 = PrimaryEpigraphConstraint(ops,
                                   objective_type=objective_type,
                                   verbose=True)
    g2 = EigenvectorEpigraphConstraint(ops, con_type='vector', eps=g_vec_eps)

    opt = EpigraphOptimizer(nlopt.LD_MMA, x.size)
    opt.set_param('dual_ftol_rel', 1e-8)
    opt.set_min_objective(f)
    opt.add_inequality_mconstraint(g1, np.zeros(g1.n_constraints))
    # opt.add_inequality_mconstraint(g2, np.zeros(g2.n_constraints))

    opt.set_lower_bounds(np.append(np.zeros(x.size-1), -np.inf))
    opt.set_upper_bounds(np.append(np.ones(x.size-1), np.inf))

    # ===== Warm start =====
    logger.info(10*'=' + "Beginning warming start" + 10*'=')
    opt.set_maxeval(starting_epoch_duration)
    x[:] = opt.optimize(x)
    ops.opt_plot.draw()
    x_history.append(x.copy())
    opt.set_ftol_rel(1e-6)
    opt.set_xtol_rel(1e-6)
    opt.set_maxeval(epoch_duration)
    logger.info(10*'=' + "Warming completed" + 10*'=')

    # ===== End Optimizer setup ======

    # ===== Optimization Loop =====
    for m in range(n_epochs):
        logger.info(10*'=' + f"Epoch {m}" + 10*'=')
        if m > 0:
            logger.info(f"===== Vector eps: {g2.eps} =====")
        for n, beta in enumerate(betas, 1):
            logger.info(f"===== Beta: {beta} ({n}/{len(betas)}) =====")
            ops.beta, ops.epoch = beta, n
            try:
                x[:] = opt.optimize(x)
            except Exception as e:
                logger.error(f"Exception occured during optimization: {e}")
                raise e
            x_history.append(x.copy())
            logger.info(
                f"Optimizer terminated with code: {opt.last_optimize_result()}")
            ops.opt_plot.draw()
            opt.set_maxeval(epoch_duration)

        print_epoch_summary(opt, m)

        if m == 0:
            opt.add_inequality_mconstraint(g2, np.zeros(g2.n_constraints))
        else:
            g2.eps = g2.eps / 2. if tighten_vector_constraint else g2.eps

    # ===== End Optimization Loop =====
    plt.show(block=True)

    # ===== Post-Processing =====
    save_final_results(ex,
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
