import inspect
from functools import partial

import fenics as fe
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
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


def plot_gradients(grad_analytical, grad_fd, title=None):
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    ax[0].plot(grad_analytical.flatten(), label='Analytical', marker='o')
    ax[0].plot(grad_fd.flatten(), label='Finite Difference', marker='.')
    ax[0].legend()
    diff = np.abs(grad_analytical - grad_fd).flatten()
    ax[1].plot(diff, label='Abs Difference')
    ax[1].legend()
    ax[2].plot(diff / np.abs(grad_analytical).flatten(),
               label='Relative Difference')
    if title:
        fig.suptitle(title)
    plt.show(block=False)

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

    if plot:
        plot_gradients(grad_analytical, grad_fd, title=constraint_name)

    diff = np.abs(grad_analytical - grad_fd)
    print(f"Max Abs Diff: {np.max(diff)}")
    print(f"Mean Abs Diff: {np.mean(diff)}")
    print(
        f"Scaled Norm Diff: {np.linalg.norm(diff)*ops.metamaterial.cell_vol}")
    try:
        npt.assert_allclose(grad_analytical, grad_fd, rtol=1e-5, atol=1e-8)
    except Exception as e:
        print(
            f"OptimizationComponent {constraint_name} finite difference check failed :(")
        print(e)

    return constraint


def initialize_filter(norm_filter_radius, metamate):
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

    return filt, filt_fn


def main():
    plt.ion()
    # ===== INPUT PARAMS =====
    E_max, E_min, nu = 1., 1/30., 0.4
    beta, eta = 1., 0.5
    norm_filter_radius = 0.1

    nelx = 5
    nely = nelx
    mesh_type = 'tri'

    basis_v = "BULK"
    extremal_mode = 1
    # ===== END INPUT PARAMS =====

    # ===== SETUP =====
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
    metamate.finite_difference_check()

    filt, filt_fn = initialize_filter(norm_filter_radius, metamate)
    ops = OptimizationState(basis_v=V,
                            extremal_mode=extremal_mode,
                            metamaterial=metamate,
                            filt=filt,
                            filt_fn=filt_fn,
                            beta=beta,
                            eta=eta,
                            show_plot=False)

    # Initial design variables
    x = np.random.uniform(0., 1, size=metamate.R.dim())
    # ===== END SETUP =====

    # ===== SCALAR CONSTRAINTS =====
    rsc = handle_constraints('RayleighScalarObjective', ops, x)
    handle_constraints('EigenvectorConstraint', ops, x, obj=rsc)
    handle_constraints('SameLargeValueConstraint', ops, x, obj=rsc)

    plt.show(block=True)


if __name__ == "__main__":
    main()
