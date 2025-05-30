import jax
import nlopt
import numpy as np
from loguru import logger as loggeru
from matplotlib import pyplot as plt
from sacred import Experiment

from experiments.utils import *
from metatop import V_DICT
from metatop.filters import setup_filter
from metatop.Metamaterial import setup_metamaterial
from metatop.optimization import OptimizationState
from metatop.optimization.scalar import (MatrixMatchingConstraint,
                                         TraceConstraint, VolumeObjective)
from metatop.profiling import ProfileConfig
from metatop.utils import beta_function, mirror_density

jax.config.update("jax_enable_x64", True)


file_name = os.path.basename(__file__)
log_file = f"{file_name}.log"
loggeru.configure(
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
loggeru.info(f"{file_name} started")

np.set_printoptions(precision=4)

# use if we want to connect to the AWS db
# load_dotenv()

ex = Experiment('extremal')


@ex.config
def config():
    E_max, E_min, nu = 1., 1./30., 0.4
    start_beta, n_betas = 1, 7
    epoch_duration, warm_start_duration = 200, 50
    extremal_mode = 1
    basis_v = 'BULK'
    dist_type = 'fro'
    nelx = nely = 50
    mirror_axis = None
    norm_filter_radius = 0.1
    show_plot = True
    vector_constraint = True
    tighten_vector_constraint = True
    g_vec_eps = 1.
    enable_profiling = False
    log_to_db = True
    verbose = True
    rtol = 1e-4
    volume_constraint = 0.


@ex.config_hook
def startup(config, command_name, logger):
    # connecto to MongoDB
    if config['log_to_db']:
        ex.observers.append(setup_mongo_observer(MONGO_URI, MONGO_DB_NAME))
    else:
        loggeru.info("MongoDB logging is diabled for this run")

    ProfileConfig.enabled = config['enable_profiling']

    return config


def setup_optimizer(objective_function, x_size, opt_type=nlopt.LD_MMA):
    """Set up the NLOpt optimizer with design variable bounds [0, 1]

    Returns:
        opt: A minimal NLOpt optimizer object without constraints

    """
    opt = nlopt.opt(opt_type, x_size)
    opt.set_min_objective(objective_function)
    opt.set_lower_bounds(0.)
    opt.set_upper_bounds(1.)
    return opt


def run_warm_start(opt: nlopt.opt, x: np.ndarray):
    """Execute the warm start phase of optimization. 

    Returns:
        x: The numpy array of the optimized result

    We run a warm start because the gradient tends to be small at the very beginning. This means that it may falsely trigger a successful termination based on relative tolerance settings. So we run first without tolerances to force it to find at least some inital design, then turn on the tolerances
    """

    x_ = x.copy()
    maxeval = opt.get_maxeval()
    if maxeval <= 0:
        loggeru.info("Warm start being skipped")
        return x_

    loggeru.info(
        f"Beginning warm start with {opt.get_maxeval()} iterations")
    try:
        x_[:] = opt.optimize(x_)
    except nlopt.ForcedStop as e:
        loggeru.error(f"NLOpt forced stop: {e}")
        raise
    except Exception as e:
        loggeru.error(f"Unexpected error when running warm start: {e}")
        raise
    loggeru.info(
        f"Warm start complete. Now shifting to graduated beta increase.")

    return x_


def run_optimization_loop(opt: nlopt.opt, x: np.ndarray, betas: list[int], ops: OptimizationState):
    """Run the main optimization loop with graduated beta increases.

    x_history: Doesn't track the incoming density, only tracks the output of the opt.optimize(x) call

    Returns:
        tuple: (x, x_history, status_code) where status_code is the final NLOpt result code.

    """
    x_ = x.copy()
    x_history = []
    loggeru.info(
        f"Running {len(betas)} beta increasing epochs with max {opt.get_maxeval()} iterations per epoch.")

    # default value, means it didn't run because NLOpt doesn't use 0
    status_code = 0

    for n, beta in enumerate(betas, 1):
        loggeru.info(f"===== Beta: {beta} ({n}/{len(betas)}) =====")
        ops.beta, ops.epoch = beta, n
        try:
            x_[:] = opt.optimize(x_)
        except nlopt.ForcedStop as e:
            loggeru.error(f"NLOpt forced stop: {e}")
        except Exception as e:
            loggeru.error(f"Unexpected error when running optimization: {e}")

        status_code = opt.last_optimize_result()
        last_value = opt.last_optimum_value()
        loggeru.info(f"Optimizer exited with code: {status_code}")
        loggeru.info(f"Final optimized value from this epoch: {last_value}")

        x_history.append(x_.copy())
        ops.epoch_iter_tracker.append(len(ops.evals))

        ops.x = x_.copy()
        save_intermediate_results(ex, ops, last_value, n)

    return x_, x_history, status_code, last_value


@ex.automain
def main(E_max, E_min, nu, start_beta, n_betas, epoch_duration, warm_start_duration, extremal_mode, basis_v, dist_type, nelx, nely, mirror_axis, norm_filter_radius, show_plot, enable_profiling, log_to_db, verbose, rtol, volume_constraint, seed):

    loggeru.debug(f"Seed: {seed}")

    betas = [start_beta * 2 ** i for i in range(n_betas)]
    # ===== Component Setup =====
    metamate = setup_metamaterial(E_max, E_min, nu, nelx, nely)

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
                            show_plot=show_plot,
                            verbose=verbose,
                            # eval_axis_kwargs=dict(yscale='log')
                            )

    # If we have a volume constriant we apply a beta function to have the mean be approximately that level so we start in a feasible spot
    if volume_constraint > 0.:
        x = beta_function(volume_constraint, size=metamate.R.dim())
    # else we just seed it randomly with approx 0.5 volume fraction
    else:
        x = np.random.uniform(0., 1., size=metamate.R.dim())

    if 'NSC' not in basis_v:
        try:
            x = mirror_density(x, metamate.R, axis=mirror_axis)[0]
        except Exception as e:
            loggeru.error(
                f"Issue with applying mirror density. Mirror not applied.")
            loggeru.error(e)
    else:
        logger.warning(
            "NSC asked for in basis_v, but so was mirror. Skipping mirroring step.")
    x = np.clip(x, a_min=1e-3, a_max=None)
    ops.x_history.append(x.copy())

    # ===== End Component Setup =====

    # ===== Optimizer setup ======
    f = VolumeObjective(ops, verbose=True)
    opt = setup_optimizer(f, x.size, opt_type=nlopt.LD_AUGLAG)
    opt.add_inequality_constraint(
        MatrixMatchingConstraint(ops, low_val=E_min, eps=1e-4, dist_type=dist_type), 0.)
    local_opt = nlopt.opt(nlopt.LD_MMA, x.size)
    opt.set_local_optimizer(local_opt)
    opt.set_xtol_rel(rtol)
    # ===== End Optimizer setup ======

    # ===== Warm start =====
    opt.set_maxeval(warm_start_duration)
    x[:] = run_warm_start(opt, x)
    ops.x_history.append(x.copy())
    ops.x = x.copy()
    save_intermediate_results(ex, ops, opt.last_optimum_value(), 0)

    # ===== Optimization Loop =====
    # Update tolerances and durations after warm start
    opt.set_maxeval(epoch_duration)
    opt.set_ftol_rel(rtol)
    opt.set_xtol_rel(rtol)
    x, x_history, status_code, final_value = run_optimization_loop(
        opt, x, betas, ops)
    ops.x_history.extend(x_history)
    ops.x = x.copy()
    loggeru.info(
        f"Optimization complete with final NLOpt status code: {status_code}")
    loggeru.info(f"Final optimized value: {final_value}")

    # ===== End Optimization Loop =====

    # ===== Post-Processing =====
    loggeru.info(calculate_elastic_constants(ops.Chom, input_style='standard'))
    try:
        save_final_results(ex, ops)
    except Exception as e:
        logger.error(f"Error when saving results: {e}")
