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
from metatop.optimization.scalar import MatrixMatchingObjective
from metatop.profiling import ProfileConfig
from metatop.utils import mirror_density

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
    E_max, E_min, nu = 1., 1/30., 0.4
    start_beta, n_betas = 8, 4
    epoch_duration, warm_start_duration = 200, 100
    extremal_mode = 1
    basis_v = 'BULK'
    nelx = nely = 50
    mirror_axis = None
    norm_filter_radius = 0.1
    show_plot = True
    vector_constraint = True
    tighten_vector_constraint = True
    g_vec_eps = 1.
    enable_profiling = False
    log_to_db = True


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

    loggeru.info(
        f"Starting warm start with {opt.get_maxeval()} iterations")
    try:
        x[:] = opt.optimize(x)
    except nlopt.ForcedStop as e:
        loggeru.error(f"NLOpt forced stop: {e}")
        raise
    except Exception as e:
        loggeru.error(f"Unexpected error when running warm start: {e}")
        raise
    loggeru.info(
        f"Warm start complete. Now shifting to graduated beta increase.")
    return x


def run_optimization_loop(opt: nlopt.opt, x: np.ndarray, betas: list[int], ops: OptimizationState, x_history: list[np.ndarray]):
    """Run the main optimization loop with graduated beta increases.

    Returns:
        tuple: (x, x_history, status_code) where status_code is the final NLOpt result code.

    """
    loggeru.info(
        f"Running {len(betas)} beta increasing epochs with max {opt.get_maxeval()} iterations per epoch.")

    # default value, means it didn't run because NLOpt doesn't use 0
    status_code = 0

    for n, beta in enumerate(betas, 1):
        loggeru.info(f"===== Beta: {beta} ({n}/{len(betas)}) =====")
        ops.beta, ops.epoch = beta, n
        try:
            x[:] = opt.optimize(x)
        except nlopt.ForcedStop as e:
            loggeru.error(f"NLOpt forced stop: {e}")
        except Exception as e:
            loggeru.error(f"Unexpected error when running optimization: {e}")

        status_code = opt.last_optimize_result()
        loggeru.info(f"Optimizer exited with code: {status_code}")

        x_history.append(x.copy())
        ops.epoch_iter_tracker.append(len(ops.evals))

    return x, x_history, status_code


@ex.automain
def main(E_max, E_min, nu, start_beta, n_betas, epoch_duration, warm_start_duration, extremal_mode, basis_v, nelx, nely, mirror_axis, norm_filter_radius, show_plot, enable_profiling, log_to_db, seed):

    run_id, outname = generate_output_filepath(
        ex, extremal_mode, basis_v, seed)

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
                            show_plot=show_plot
                            # eval_axis_kwargs=dict(yscale='log')
                            )

    x = np.random.uniform(0., 1., size=metamate.R.dim())
    try:
        x = mirror_density(x, metamate.R, axis=mirror_axis)[0]
    except Exception as e:
        loggeru.error(
            f"Issue with applying mirror density. Mirror not applied.")
        loggeru.error(e)
    x_history = [x.copy()]

    # ===== End Component Setup =====

    # ===== Optimizer setup ======
    f = MatrixMatchingObjective(ops, low_val=E_min)
    opt = setup_optimizer(f, x.size)
    # ===== End Optimizer setup ======

    # ===== Warm start =====
    opt.set_maxeval(warm_start_duration)
    x = run_warm_start(opt, x)

    # ===== Optimization Loop =====
    # Update tolerances and durations after warm start
    opt.set_maxeval(epoch_duration)
    opt.set_ftol_rel(1e-6)
    opt.set_xtol_rel(1e-6)
    x, x_history, status_code = run_optimization_loop(
        opt, x, betas, ops, x_history)

    loggeru.info(
        f"Optimization complete with final NLOpt status code: {status_code}")

    # ===== End Optimization Loop =====

    # ===== Post-Processing =====
    Chom = f.forward(x)[1]
    loggeru.info(calculate_elastic_constants(Chom, input_style='standard'))
    # save_results(ex,
    #              run_id, t:
    #              outname,
    #              metamate,
    #              img_rez,
    #              img_shape,
    #              ops,
    #              x,
    #              f,
    #              x_history)

    # if f.show_plot:
    #     plt.close(f.fig)
