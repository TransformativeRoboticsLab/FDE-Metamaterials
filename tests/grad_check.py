import inspect
from functools import partial

import fenics as fe
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
from loguru import logger
from tqdm import tqdm

import metatop.optimization.epigraph as epi
import metatop.optimization.scalar as sca
from metatop import V_DICT
from metatop.filters import (DensityFilter, HelmholtzFilter,
                             jax_density_filter, jax_helmholtz_filter)
from metatop.Metamaterial import setup_metamaterial
from metatop.optimization import OptimizationState
from metatop.optimization.OptimizationComponents import (
    OptimizationComponent, ScalarOptimizationComponent,
    VectorOptimizationComponent)

jax.config.update("jax_enable_x64", True)


np.random.seed(0)


class QuadraticChecker(ScalarOptimizationComponent):

    def __call__(self, x, grad):

        c, dc = jax.value_and_grad(self.eval)(x)

        if grad.size > 0:
            grad[:] = dc

        return float(c)

    def eval(self, x):
        return jnp.sum(x**2)

    def adjoint(self):
        pass

    def analytical_grad(self, x):
        return 2 * x


def precompute_obj(obj, x):
    '''
    Precompute the objective the if the constraint relies on calling the objective first.
    The objective could be an nlopt scalar or vector so we need to handle how many arguments it takes dynamically.
    '''
    arg_nums = len(inspect.signature(obj).parameters)
    empty_grad = np.array([])
    if arg_nums > 2:
        obj(np.zeros(obj.n_constraints), x, empty_grad)
    elif arg_nums == 2:
        obj(x, empty_grad)
    else:
        raise ValueError("Invalid number of arguments for primary objective")


def finite_difference_checker(constraint: OptimizationComponent, x, obj=None, epsilon=1e-6, name=None):
    # args_count = len(inspect.signature(constraint).parameters)
    # grad_fd = np.zeros(x.size) if args_count == 2 else np.zeros(
    # (constraint.n_constraints, x.size))
    if isinstance(constraint, ScalarOptimizationComponent):
        grad_fd = np.zeros(x.size)
    elif isinstance(constraint, VectorOptimizationComponent):
        grad_fd = np.zeros((constraint.n_constraints, x.size))
    else:
        raise ValueError(
            f"Constraint {constraint} must be derived from ScalarOptimizationComponent or VectorOptimizationComponent")

    for i in tqdm(range(grad_fd.shape[grad_fd.ndim-1]), desc=f"Checking gradient {name}"):
        perturb = np.zeros_like(x)
        perturb[i] = epsilon
        x_plus = x + perturb
        x_minus = x - perturb

        empty_grad = np.array([])

        if isinstance(constraint, ScalarOptimizationComponent):
            precompute_obj(obj, x_plus) if obj is not None else None
            c_plus = constraint(x_plus, empty_grad)

            precompute_obj(obj, x_minus) if obj is not None else None
            c_minus = constraint(x_minus, empty_grad)

            grad_fd[i] = (c_plus - c_minus) / (2 * epsilon)
        else:
            results_plus = np.zeros(constraint.n_constraints)
            results_minus = np.zeros(constraint.n_constraints)

            precompute_obj(obj, x_plus) if obj is not None else None
            constraint(results_plus, x_plus, empty_grad)

            precompute_obj(obj, x_minus) if obj is not None else None
            constraint(results_minus, x_minus, empty_grad)

            grad_fd[:, i] = (results_plus - results_minus) / (2 * epsilon)

    return grad_fd


def plot_gradients(actual: np.ndarray, desired: np.ndarray, title: str, rtol: float, atol: float):
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(10, 10))

    act = actual.flatten()
    des = desired.flatten()

    abs_diff = np.abs(des-act)
    rel_diff = abs_diff / np.abs(des + 1e-6)

    # Actual values plot
    ax0.plot(act, label='Analytical', marker='o')
    ax0.plot(des, label='Finite Difference', marker='.')
    ax0.legend()

    # Absolute diff plot
    ax1.plot(abs_diff)
    ax1.axhline(atol, color='k', lw=2, alpha=0.5) if atol else None
    ax1.set(title="Absolute Difference",
            yscale='log')
    # Relative diff plot
    ax2.plot(rel_diff)
    ax2.axhline(rtol, color='k', lw=2, alpha=0.5) if rtol else None
    ax2.set(title="Relative Difference",
            yscale='log')

    # np.allclose check
    ax3.plot(abs_diff, label='|actual - desired|')
    ax3.plot(atol + rtol * np.abs(des), label='(atol + rtol * |desired|)')
    ax3.set(title="Check |actual - desired| <= (atol + rtol * |desired|)",
            yscale='log')
    ax3.legend()
    if title:
        fig.suptitle(title)
    plt.draw()
    plt.pause(1e-3)

    return fig


def create_constraint(cname, config):
    constraint_list = {
        'EpigraphObjective': epi.EpigraphObjective,
        'PrimaryEpigraphConstraint': epi.PrimaryEpigraphConstraint,
        'EigenvectorEpigraphConstraint': epi.EigenvectorEpigraphConstraint,
        'RayleighScalarObjective': sca.RayleighRatioObjective,
        'EigenvectorConstraint': sca.EigenvectorConstraint,
        'SameLargeValueConstraint': sca.SameLargeValueConstraint,
        'VolumeObjective': sca.VolumeObjective,
        'MatrixMatchingConstraint': sca.MatrixMatchingConstraint,
    }
    if cname not in constraint_list:
        s = f"{cname} not a valid constraint name. List is [{constraint_list.keys()}]"
        raise ValueError(s)

    return constraint_list[cname](**config)


def _create_optimization_component(name, config):
    """Helper function to create an optimization component."""

    try:
        return create_constraint(name, config)
    except ValueError as e:
        logger.error(f"Unable to create {name}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error creating {name}: {e}")

    return None


def _calculate_analytical_gradient(component, x):
    """Calculate analytical gradient for the given component type."""
    if isinstance(component, ScalarOptimizationComponent):
        grad_analytical = np.zeros(x.size)
        component(x, grad_analytical)
        return grad_analytical
    elif isinstance(component, VectorOptimizationComponent):
        grad_analytical = np.zeros((component.n_constraints, x.size))
        results = np.zeros(component.n_constraints)
        component(results, x, grad_analytical)
        return grad_analytical
    else:
        raise ValueError(f"Unknown component type: {type(component)}")


def handle_optimization_component(name: str, ops: OptimizationState, x: np.ndarray, plot: bool = True, obj: OptimizationComponent = None, **kwargs):
    """
    Create, validate and test an optimization component with finite difference gradient checking.

    Args:
        name: Class name of the optimization component to create
        ops: Optimization state object containing problem setup
        x: Design variables
        epi_constraint: If True, append a DOF for the t variable (epigraph form)
        plot: If True, plot gradient comparison
        obj: Primary objective function that might need to be evaluated first

    Returns:
        The validated optimization component or None if validation failed
    """
    # Create the component
    component_config = {'ops': ops, **kwargs}
    component = _create_optimization_component(name, component_config)
    if component is None:
        return None

    # Handle epigraph form if needed
    if isinstance(component, epi.EpigraphComponent):
        x = np.append(x, 1.)
        logger.info(f"{component} is an epigraph component. Appending t to x")

    # Run primary objective if provided and valid
    if isinstance(obj, OptimizationComponent):
        precompute_obj(obj, x)
    elif obj is not None:
        logger.warning(
            "Argument obj is not an OptimizationComponent and will be ignored")

    # Calculate analytical gradient
    try:
        grad_analytical = _calculate_analytical_gradient(component, x)
    except ValueError as e:
        logger.error(f"Failed to calculate analytical gradient: {e}")
        return None

    # Calculate finite difference gradient
    grad_fd = finite_difference_checker(component, x, obj, name=name)

    # Validate gradients
    rtol, atol = 1e-5, 1e-8
    if plot:
        plot_gradients(grad_analytical, grad_fd,
                       title=name, rtol=rtol, atol=atol)

    try:
        npt.assert_allclose(grad_analytical, grad_fd, rtol=rtol, atol=atol)
        logger.info(f"PASS: Component {component}")
        return component
    except Exception as e:
        logger.error(
            f"FAIL: Component {component}: {e}")
        return None


def initialize_filter(norm_filter_radius, metamate):
    if metamate.R.ufl_element().degree() > 0:
        logger.info("Using Helmholtz filter")
        filt = HelmholtzFilter(radius=norm_filter_radius,
                               fn_space=metamate.R)
        filt_fn = partial(jax_helmholtz_filter, filt)
    elif metamate.R.ufl_element().degree() == 0:
        logger.info("Using Density filter")
        filt = DensityFilter(mesh=metamate.mesh,
                             radius=norm_filter_radius,
                             distance_method='periodic')
        filt_fn = partial(jax_density_filter, filt.H_jax, filt.Hs_jax)
    else:
        raise ValueError(
            "Invalid filter type. Must be DensityFilter or HelmholtzFilter")

    return filt, filt_fn


def self_check():

    quad_check = QuadraticChecker(OptimizationState(show_plot=False))

    N = 100
    x = np.random.normal(size=N)
    grad_jax = np.zeros_like(x)

    quad_check(x, grad_jax)
    grad_fd = finite_difference_checker(quad_check, x)

    grad_an = quad_check.analytical_grad(x)

    try:
        npt.assert_allclose(grad_jax, grad_fd, rtol=1e-5, atol=1e-8)
        npt.assert_allclose(grad_an, grad_fd, rtol=1e-5, atol=1e-8)
    except AssertionError as e:
        logger.error(
            f"Self check on finite difference method did not pass: {e}")

    logger.info("PASS: Self check of finite differencing")


def main():
    self_check()
    plt.ion()
    # ===== INPUT PARAMS =====
    E_max, E_min, nu = 1., 1/30., 0.4
    beta, eta = 1., 0.5
    norm_filter_radius = 0.1

    nelx = 6
    nely = nelx
    mesh_type = 'tri'

    basis_v = None
    extremal_mode = 1
    # ===== END INPUT PARAMS =====

    # ===== SETUP =====
    check_metamaterial = False
    check_filter = False
    check_scalars = False
    check_epigraphs = False
    check_volume = True
    if basis_v:
        V = V_DICT[basis_v]
    else:
        # create random orthonormal matrix
        random_matrix = np.random.randn(3, 3)
        V = np.linalg.qr(random_matrix)[0]

    metamate = setup_metamaterial(E_max,
                                  E_min,
                                  nu,
                                  nelx,
                                  nely,
                                  mesh_cell_type=mesh_type,
                                  domain_shape='square')

    filt, filt_fn = initialize_filter(norm_filter_radius, metamate)
    ops = OptimizationState(basis_v=V,
                            extremal_mode=extremal_mode,
                            metamaterial=metamate,
                            filt=filt,
                            filt_fn=filt_fn,
                            beta=beta,
                            eta=eta,
                            show_plot=True,
                            verbose=False,
                            silent=True)

    # Initial design variables
    x = np.random.uniform(0., 1, size=metamate.R.dim())
    config = dict(
        ops=ops,
        x=x,
        plot=False
    )
    # ===== END SETUP =====

    # ===== CHECK COMPONENTS =====
    if check_metamaterial:
        try:
            metamate.check_gradient()
        except Exception as e:
            logger.error(e)
    else:
        logger.warning("Skipping metamaterial gradient check")

    if check_filter:
        filt.check_gradient(x)
    else:
        logger.warning("Skipping filter gradient check")

    # ===== SCALAR CONSTRAINTS =====
    if check_scalars:
        logger.info("Checking scalar components")
        rsc = handle_optimization_component(
            'RayleighScalarObjective', **config)
        handle_optimization_component(
            'EigenvectorConstraint', **config, obj=rsc)
        handle_optimization_component(
            'SameLargeValueConstraint', **config, obj=rsc)
    else:
        logger.warning("Skipping scalar component checks")

    if check_epigraphs:
        logger.info("Checking epigraph components")
        handle_optimization_component('EpigraphObjective', **config)
        pec = handle_optimization_component(
            'PrimaryEpigraphConstraint', **config, objective_type='ray')
        eps = 1.
        handle_optimization_component(
            'EigenvectorEpigraphConstraint', **config, con_type='scalar', eps=eps, obj=pec)
        handle_optimization_component(
            'EigenvectorEpigraphConstraint', **config, con_type='vector', eps=eps, obj=pec)
    else:
        logger.warning("Skipping epigraph component checks")

    if check_volume:
        logger.info("Checking volume constraint")
        obj = handle_optimization_component('VolumeObjective', **config)
        handle_optimization_component(
            'MatrixMatchingConstraint', obj=obj, **config)


if __name__ == "__main__":
    main()
    plt.show(block=True)
