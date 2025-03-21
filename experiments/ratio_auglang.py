import incense.artifact
import jax
import nlopt
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from sacred import Experiment

from experiments.utils import *
from metatop import V_DICT
from metatop.filters import setup_filter
from metatop.Metamaterial import setup_metamaterial
from metatop.optimization import OptimizationState
from metatop.optimization.scalar import (NormEigenvectorConstraint,
                                         RayleighRatioObjective,
                                         SameLargeValueConstraint)
from metatop.profiling import ProfileConfig

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
         "level": "INFO",
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
    nelx = nely = 50
    norm_filter_radius = 0.1
    verbose = False
    interim_plot = True
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
        print("MongoDB logging is diabled for this run")

    ProfileConfig.enabled = config['enable_profiling']

    return config


@ex.automain
def main(E_max, E_min, nu, start_beta, n_betas, n_epochs, epoch_duration, starting_epoch_duration, extremal_mode, basis_v, nelx, nely, norm_filter_radius, verbose, interim_plot, vector_constraint, tighten_vector_constraint, g_vec_eps, enable_profiling, log_to_db, seed):

    run_id, outname = generate_output_filepath(
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
                            plot_interval=10,
                            )

    # x = np.random.choice([0, 1], size=metamate.R.dim())
    x = np.random.uniform(0., 1., size=metamate.R.dim())
    # x = mirror_density(x, metamate.R, type='xy')[0]

    # ===== End Component Setup =====

    # ===== Optimizer setup ======
    f = RayleighRatioObjective(ops, verbose=True)
    g1 = EigenvectorConstraint(ops, eps=1e-3, verbose=True)
    g2 = SameLargeValueConstraint(ops, eps=1e-3, verbose=True)

    opt = nlopt.opt(nlopt.LD_AUGLAG, x.size)
    opt.set_min_objective(f)

    local_opt = nlopt.opt(nlopt.LD_MMA, x.size)
    opt.set_local_optimizer(local_opt)

    opt.add_inequality_constraint(g1, 0.)
    opt.add_inequality_constraint(g2, 0.)

    opt.set_lower_bounds(0.)
    opt.set_upper_bounds(1.)
    # opt.set_maxeval(starting_epoch_duration)
    # ===== End Optimizer setup ======

    # ===== Optimization Loop =====
    x_history = [x.copy()]
    for n, beta in enumerate(betas, 1):
        print(f"===== Beta: {beta} ({n}/{len(betas)}) =====")
        ops.beta, ops.epoch = beta, n
        x[:] = opt.optimize(x)
        # try:
        #     x[:] = opt.optimize(x)
        # except nlopt.ForcedStop:
        #     print(f"nlopt forced stop {e}")
        #     # sys.exit(-2)
        # except Exception as e:
        #     print(f"Optimization stopped {e}")
        #     # sys.exit(-1)
        x_history.append(x.copy())
        opt.set_maxeval(epoch_duration)
        ops.epoch_iter_tracker.append(len(ops.evals))

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
                 f,
                 x_history)

    if f.show_plot:
        plt.close(f.fig)
