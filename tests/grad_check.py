import inspect
from functools import partial

import fenics as fe
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
from tqdm import tqdm

from metatop import V_DICT
from metatop.filters import (DensityFilter, HelmholtzFilter,
                             jax_density_filter, jax_helmholtz_filter)
from metatop.metamaterial import Metamaterial
from metatop.optimization import OptimizationState
from metatop.optimization.epigraph import (EigenvalueProblemConstraints,
                                           EigenvectorConstraint,
                                           EpigraphObjective,
                                           ExtremalConstraints,
                                           InvariantsConstraint,
                                           SpectralNormConstraint,
                                           TraceConstraint, VolumeConstraint)
from metatop.optimization.examples import AndreassenOptimization
from metatop.optimization.scalar import \
    BulkModulusConstraint as ScalarBulkModulusConstraint
from metatop.optimization.scalar import RayleighScalarObjective

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


def finite_difference_checker(constraint, x, obj=None, epsilon=1e-6, name=None):
    args_count = len(inspect.signature(constraint).parameters)
    grad_fd = np.zeros(x.size) if args_count == 2 else np.zeros(
        (constraint.n_constraints, x.size))

    for i in tqdm(range(grad_fd.shape[grad_fd.ndim-1]), desc=f"Checking gradient {name}"):
        perturb = np.zeros_like(x)
        perturb[i] = epsilon
        x_plus = x + perturb
        x_minus = x - perturb

        empty_grad = np.array([])

        if args_count > 2:  # fn(results, x, grad)
            r_plus = np.zeros(constraint.n_constraints)
            r_minus = np.zeros(constraint.n_constraints)

            precompute_obj(obj, x_plus) if obj is not None else None
            constraint(r_plus, x_plus, empty_grad)

            precompute_obj(obj, x_minus) if obj is not None else None
            constraint(r_minus, x_minus, empty_grad)

            grad_fd[:, i] = (r_plus - r_minus) / (2 * epsilon)
        elif args_count == 2:  # c = fn(x, grad)
            precompute_obj(obj, x_plus) if obj is not None else None
            c_plus = constraint(x_plus, empty_grad)

            precompute_obj(obj, x_minus) if obj is not None else None
            c_minus = constraint(x_minus, empty_grad)

            grad_fd[i] = (c_plus - c_minus) / (2 * epsilon)
        else:
            raise ValueError(
                "Invalid number of arguments for primary objective.")

    return grad_fd


def plot_gradients(grad_analytical, grad_fd):
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    ax[0].plot(grad_analytical.flatten(), label='Analytical', marker='o')
    ax[0].plot(grad_fd.flatten(), label='Finite Difference', marker='.')
    ax[0].legend()
    diff = np.abs(grad_analytical - grad_fd).flatten()
    ax[1].plot(diff, label='Abs Difference')
    ax[1].legend()
    ax[2].plot(diff / np.abs(grad_analytical).flatten(),
               label='Relative Difference')
    plt.show(block=False)

    return fig


def create_constraint(cname, p):
    if cname == 'Epigraph':
        return EpigraphObjective()
    elif cname == 'EigenvalueProblem':
        return EigenvalueProblemConstraints(basis_v=p['v'],
                                            extremal_mode=p['extremal_mode'],
                                            metamaterial=p['metamaterial'], ops=p['ops'],
                                            verbose=p['verbose'],
                                            show_plot=p['plot'],
                                            objective_type=p['objective_type'],
                                            silent=['silent'])
    elif cname == 'Extremal':
        return ExtremalConstraints(basis_v=p['v'],
                                   extremal_mode=p['extremal_mode'],
                                   metamaterial=p['metamaterial'], ops=p['ops'],
                                   verbose=p['verbose'],
                                   show_plot=p['plot'],
                                   objective_type=p['objective_type'],
                                   silent=p['silent'])
    elif cname == 'Energy':
        return RayleighScalarObjective(v=p['v'],
                                       extremal_mode=p['extremal_mode'],
                                       metamaterial=p['metamaterial'], ops=p['ops'],
                                       verbose=p['verbose'],
                                       plot=p['plot'])
    elif cname == 'Invariants':
        return InvariantsConstraint(ops=p['ops'],
                                    verbose=p['verbose'])
    elif cname == 'Eigenvector':
        return EigenvectorConstraint(basis_v=p['v'],
                                     ops=p['ops'],
                                     verbose=p['verbose'])
    elif cname == 'SpectralNorm':
        return SpectralNormConstraint(ops=p['ops'],
                                      bound=1.,
                                      verbose=p['verbose'])
    elif cname == 'Trace':
        return TraceConstraint(ops=p['ops'], bound=1., verbose=p['verbose'])

    elif cname == 'Volume':
        return VolumeConstraint(ops=p['ops'], bound=0.5, verbose=p['verbose'])
    elif cname == 'ScalarBulk':
        return ScalarBulkModulusConstraint(
            base_E=p['metamaterial'].prop.E_max,
            base_nu=p['metamaterial'].prop.nu,
            a=0.2*1e-2,
            ops=p['ops'],
            verbose=p['verbose'])

    # elif constraint_name == 'ScalarVolume':
    #     return ScalarVolumeConstraint(
    #         V=0.35,
    #         ops=params['ops'],
    #         verbose=params['verbose'])
    # elif constraint_name == 'ScalarIsotropic':
    #     return ScalarIsotropicConstraint(
    #         eps=1e-5,
    #         ops=params['ops'],
    #         verbose=params['verbose'])
    elif cname == 'Andreassen':
        return AndreassenOptimization('pr',
                                      metamaterial=p['metamaterial'],
                                      ops=p['ops'],
                                      verbose=p['verbose'],
                                      plot=p['plot'])
    else:
        raise ValueError(f"Constraint {cname} not found")


def get_analytical_gradient(constraint, x):
    arg_nums = len(inspect.signature(constraint).parameters)
    if arg_nums > 2:
        grad_analytical = np.zeros((constraint.n_constraints, x.size))
        constraint(np.zeros(constraint.n_constraints), x, grad_analytical)
    elif arg_nums == 2:
        grad_analytical = np.zeros(x.size)
        constraint(x, grad_analytical)
    else:
        raise ValueError("Invalid number of arguments")
    return grad_analytical


def handle_constraints(constraint_name, x, params, epi_constraint=False, plot=False):
    # if the constraint is formulated for an epigraph form we need to add a DOF for the t variable
    if epi_constraint:
        x = np.append(x, 1.)

    # if we need to run the primary objective function first to get the analytical gradient
    obj = params['obj']
    precompute_obj(obj, x) if obj is not None else None

    constraint = create_constraint(constraint_name, params)
    grad_analytical = get_analytical_gradient(constraint, x)
    grad_fd = finite_difference_checker(
        constraint, x, obj, name=constraint_name)

    if plot:
        plot_gradients(grad_analytical, grad_fd)

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
        filter_fn = partial(jax_helmholtz_filter, filt)
    elif metamate.R.ufl_element().degree() == 0:
        print("Using Density filter")
        filt = DensityFilter(mesh=metamate.mesh,
                             radius=norm_filter_radius,
                             distance_method='periodic')
        filter_fn = partial(jax_density_filter, filt.H_jax, filt.Hs_jax)
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
