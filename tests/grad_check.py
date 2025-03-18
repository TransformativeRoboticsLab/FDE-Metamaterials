import inspect
from functools import partial

import fenics as fe
import jax
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
from metatop.metamaterial import setup_metamaterial
from metatop.optimization import OptimizationState
from metatop.optimization.utils import (OptimizationComponent,
                                        ScalarOptimizationComponent,
                                        VectorOptimizationComponent)

jax.config.update("jax_enable_x64", True)


np.random.seed(0)


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
    grad_fd = np.zeros((constraint.n_constraints, x.size))

    for i in tqdm(range(grad_fd.shape[grad_fd.ndim-1]), desc=f"Checking gradient {name}"):
        perturb = np.zeros_like(x)
        perturb[i] = epsilon
        x_plus = x + perturb
        x_minus = x - perturb

        empty_grad = np.array([])

        if constraint.n_constraints == 1:
            precompute_obj(obj, x_plus) if obj is not None else None
            c_plus = constraint(x_plus, empty_grad)

            precompute_obj(obj, x_minus) if obj is not None else None
            c_minus = constraint(x_minus, empty_grad)

            grad_fd[0, i] = (c_plus - c_minus) / (2 * epsilon)
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
    rel_diff = abs_diff / np.abs(des)

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
        'Epigraph': epi.EpigraphObjective,
        'RayleighScalarObjective': sca.RayleighRatioObjective,
        'EigenvectorConstraint': sca.EigenvectorConstraint,
        'SameLargeValueConstraint': sca.SameLargeValueConstraint,
    }
    if cname not in constraint_list:
        s = f"{cname} not a valid constraint name. List is [{constraint_list.keys()}]"
        raise ValueError(s)

    return constraint_list[cname](**config)


def handle_constraints(constraint_name: str, ops: OptimizationState, x: np.ndarray, verbose: bool = False, silent: bool = True, epi_constraint: bool = False, plot: bool = True, obj: OptimizationComponent = None):
    # if the constraint is formulated for an epigraph form we need to add a DOF for the t variable
    if epi_constraint:
        x = np.append(x, 1.)

    # if we need to run the primary objective function first to get the analytical gradient
    if isinstance(obj, OptimizationComponent):
        precompute_obj(obj, x)
    elif obj:
        print(
            "Argument obj is not an OptimizationComponent. I can't do anything with this...")

    constraint_config = {
        'ops': ops,
        'verbose': verbose,
        'silent': silent,
    }

    try:
        constraint = create_constraint(constraint_name, constraint_config)
    except ValueError as e:
        print(f"Unable to create {constraint_name}: {e}")
    except Exception as e:
        print(e)
        return

    if not constraint:
        return
    grad_analytical = np.zeros((constraint.n_constraints, x.size))
    constraint(x, grad_analytical)
    grad_fd = finite_difference_checker(
        constraint, x, obj, name=constraint_name)

    rtol, atol = 1e-5, 1e-8
    if plot:
        plot_gradients(grad_analytical, grad_fd,
                       title=constraint_name, rtol=rtol, atol=atol)

    diff = np.abs(grad_analytical - grad_fd)
    print(f"Max Abs Diff: {np.max(diff)}")
    print(f"Mean Abs Diff: {np.mean(diff)}")
    print(
        f"Scaled Norm Diff: {np.linalg.norm(diff)*params['metamaterial'].cell_vol}")
    npt.assert_allclose(grad_analytical, grad_fd, rtol=1e-5, atol=1e-5)


def main():
    nelx = 5
    nely = nelx
    E_max, E_min, nu = 1., 1e-9, 0.3
    beta, eta = 1., 0.5
    cell_side_length_mm = 25.
    line_width_mm = 2.5
    line_space_mm = line_width_mm
    norm_line_width = line_width_mm / cell_side_length_mm
    norm_line_space = line_space_mm / cell_side_length_mm
    norm_filter_radius = norm_line_width

    meshes = {
        'tri': fe.UnitSquareMesh(nelx, nely, 'crossed'),
        'quad': fe.RectangleMesh.create([fe.Point(0, 0), fe.Point(1, 1)], [
                                        nelx, nely], fe.CellType.Type.quadrilateral)
    }
    mesh_type = 'tri'

    # Generate a random 3x3 matrix
    # Use QR decomposition to get an orthonormal basis
    random_matrix = np.random.randn(3, 3)
    basis_v = np.linalg.qr(random_matrix)[0]

    # Setup parameters and initial conditions
    params = {
        'metamaterial': Metamaterial(E_max=E_max, E_min=E_min, nu=nu, nelx=nelx, nely=nely, mesh=meshes[mesh_type]),
        'ops': OptimizationState(beta=beta, eta=eta),
        'v': basis_v,
        'extremal_mode': 1,
        'verbose': False,
        'plot': False,
        'obj': None,
        'line_width': norm_line_width,
        'line_space': norm_line_space,
        'objective_type': 'ray',
        'silent': False,
    }

    metamate = params['metamaterial']
    metamate.create_function_spaces()
    metamate.initialize_variational_forms()
    if metamate.R.ufl_element().degree() > 0:
        print("Using Helmholtz filter")
        filt = HelmholtzFilter(radius=norm_filter_radius,
                               fn_space=metamate.R)
        filt_fn = partial(jax_helmholtz_filter, filt)
    elif metamate.R.ufl_element().degree() == 0:
        print("Using Density filter")
        filt = DensityFilter(mesh=metamate.mesh,
                             radius=norm_filter_radius,
                             distance_method='periodic')
        filt_fn = partial(jax_density_filter, filt.H_jax, filt.Hs_jax)
    else:
        raise ValueError(
            "Invalid filter type. Must be DensityFilter or HelmholtzFilter")

    params['ops'].filt = filt
    params['ops'].filt_fn = filter_fn
    # Initial design variables
    x = np.random.uniform(1e-3, 1, params['metamaterial'].R.dim())

    # The constraints that do not need another constraint to be run first
    # handle_constraints('Energy', x, params)
    # handle_constraints('Epigraph', x, params, epi_constraint=True)
    # handle_constraints('Extremal', x, params, epi_constraint=True)
    # handle_constraints('Andreassen', x, params, plot=True)

    handle_constraints('EigenvalueProblem',
                       np.concatenate((x, basis_v.flatten())),
                       params,
                       epi_constraint=True)

    # ===== Scalar Constraints =====
    obj = RayleighScalarObjective(basis_v=params['v'],
                                  extremal_mode=params['extremal_mode'],
                                  metamaterial=params['metamaterial'],
                                  ops=params['ops'],
                                  verbose=params['verbose'],
                                  plot=params['plot'])
    params['obj'] = obj

    # handle_constraints('ScalarBulk', x, params)
    # handle_constraints('ScalarVolume', x, params)
    # handle_constraints('ScalarIsotropic', x, params, plot=True)

    # ===== Epigraph Constraints =====
    # Now we need to run the primary objective function before each run because this is the one that does the actual FEM s        'Extremal':
    obj = ExtremalConstraints(basis_v=params['v'],
                              extremal_mode=params['extremal_mode'],
                              metamaterial=params['metamaterial'], ops=params['ops'],
                              verbose=params['verbose'],
                              show_plot=params['plot'],
                              objective_type=params['objective_type'])
    params['obj'] = obj

    # handle_constraints('Invariants', x, params, epi_constraint=True)
    # handle_constraints('Eigenvector', x, params, epi_constraint=True)
    # handle_constraints('Geometric', x, params, epi_constraint=True)
    # handle_constraints('SpectralNorm', x, params, epi_constraint=True)
    # handle_constraints('Trace', x, params, epi_constraint=True)
    # handle_constraints('Volume', x, params, epi_constraint=True)

    plt.show(block=True)


if __name__ == "__main__":
    main()
    plt.show(block=True)
