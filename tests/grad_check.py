import inspect
from functools import partial

import fenics as fe
import jax

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from metatop import V_DICT
from metatop.filters import (DensityFilter, HelmholtzFilter,
                             jax_density_filter, jax_helmholtz_filter)
from metatop.metamaterial import Metamaterial
from metatop.optimization import OptimizationState
from metatop.optimization.epigraph import (EigenvectorConstraint, Epigraph,
                                           ExtremalConstraints,
                                           InvariantsConstraint,
                                           SpectralNormConstraint,
                                           TraceConstraint, VolumeConstraint)
from metatop.optimization.scalar import EnergyObjective

np.random.seed(0)

def finite_difference_checker(constraint, x, grad_analytical, params, epsilon=1e-5):
    args_count = len(inspect.signature(constraint).parameters)
    grad_fd = np.zeros_like(grad_analytical)

    for i in tqdm(range(grad_fd.shape[grad_fd.ndim-1]), desc="Checking gradient"):
        perturb = np.zeros_like(x)
        perturb[i] = epsilon
        x_plus = x + perturb
        x_minus = x - perturb
        
        obj = params['obj']
        empty_grad = np.array([])

        if args_count > 2:  # fn(results, x, grad)
            r_plus = np.zeros(constraint.n_constraints)
            r_minus = np.zeros(constraint.n_constraints)

            obj(r_plus, x_plus, empty_grad) if not None else None
            constraint(r_plus, x_plus, empty_grad)

            obj(r_minus, x_minus, empty_grad) if not None else None
            constraint(r_minus, x_minus, empty_grad)

            grad_fd[:, i] = (r_plus - r_minus) / (2 * epsilon)
        elif args_count == 2:  # c = fn(x, grad)
            obj(x_plus, empty_grad) if not None else None
            c_plus = constraint(x_plus, empty_grad)

            obj(x_minus, empty_grad) if not None else None
            c_minus = constraint(x_minus, empty_grad)

            grad_fd[i] = (c_plus - c_minus) / (2 * epsilon)
        else:
            raise ValueError("Invalid number of arguments")

    diff = np.abs(grad_analytical - grad_fd)
    print(f"Max Abs Diff: {np.max(diff)}")
    print(f"Mean Abs Diff: {np.mean(diff)}")
    print(f"Norm Diff: {np.linalg.norm(diff)*params['metamaterial'].cell_vol}")

    plot_gradients(grad_analytical, grad_fd)

    


def plot_gradients(grad_analytical, grad_fd):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(grad_analytical.flatten(), label='Analytical', marker='o')
    ax[0].plot(grad_fd.flatten(), label='Finite Difference', marker='.')
    ax[0].legend()
    ax[1].plot(np.abs(grad_analytical - grad_fd).flatten(), label='Abs Difference')
    ax[1].legend()
    plt.show()


def handle_constraints(constraint_name, x, params, epi_constraint=False):
    constraints = {
        'Epigraph': Epigraph(),
        'Extremal': ExtremalConstraints(v=params['v'], 
                                        extremal_mode=params['extremal_mode'], 
                                        metamaterial=params['metamaterial'], ops=params['ops'], 
                                        verbose=params['verbose'], 
                                        show_plot=params['plot'],
                                        objective_type=params['objective_type']),
        'Energy': EnergyObjective(v=params['v'], 
                                   extremal_mode=params['extremal_mode'], 
                                   metamaterial=params['metamaterial'], ops=params['ops'], 
                                   verbose=params['verbose'], 
                                   plot=params['plot']),
        'Invariants': InvariantsConstraint(ops=params['ops'], 
                                           verbose=params['verbose']),
        'Eigenvector': EigenvectorConstraint(v=params['v'],
                                             ops=params['ops'],
                                             verbose=params['verbose']),
        'SpectralNorm': SpectralNormConstraint(ops=params['ops'],
                                               bound=1.,
                                               verbose=params['verbose']),
        'Trace': TraceConstraint(ops=params['ops'], bound=1., verbose=params['verbose']),
        'Volume': VolumeConstraint(ops=params['ops'], bound=0.5, verbose=params['verbose']),
    }

    # if the constraint is formulated for an epigraph form we need to add a DOF for the t variable
    if epi_constraint:
        x = np.append(x, 1.)

    # if we need to run the primary objective function first to get the analytical gradient
    if params['obj']:
        params['obj'](np.zeros(params['obj'].n_constraints), x, np.array([]), dummy_run=True)

    constraint = constraints[constraint_name]
    arg_nums = len(inspect.signature(constraint).parameters)
    if arg_nums > 2:
        grad_analytical = np.zeros((constraint.n_constraints, x.size))
        constraint(np.zeros(constraint.n_constraints), x, grad_analytical)
    elif arg_nums == 2:
        grad_analytical = np.zeros(x.size)
        constraint(x, grad_analytical)
    else:
        raise ValueError("Invalid number of arguments")
    finite_difference_checker(constraint, x, grad_analytical, params)


def main():
    nelx = 5
    nely = nelx
    E_max, E_min, nu = 1., 1e-2, 0.45
    beta, eta = 8., 0.5
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
    v_basis = 'BULK'

    # Setup parameters and initial conditions
    params = {
        'metamaterial': Metamaterial(E_max=E_max, E_min=E_min, nu=nu, nelx=nelx, nely=nely, mesh=meshes[mesh_type]),
        'ops': OptimizationState(beta=beta, eta=eta),
        'v': V_DICT[v_basis],
        'extremal_mode': 1,
        'verbose': False,
        'plot': False,
        'obj': None,  
        'line_width': norm_line_width,
        'line_space': norm_line_space,
        'objective_type': 'norm'
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
        raise ValueError("Invalid filter type. Must be DensityFilter or HelmholtzFilter")
                    
    params['ops'].filt = filt
    params['ops'].filt_fn = filter_fn
    # Initial design variables
    x = np.random.uniform(1e-3, 1, params['metamaterial'].R.dim())

    # The three constraints that do not need another constraint to be run first
    # handle_constraints('Energy', x, params)
    # handle_constraints('Epigraph', x, params, epi_constraint=True)
    # handle_constraints('Extremal', x, params, epi_constraint=True)

    # Now we need to run the primary objective function before each run because this is the one that does the actual FEM s        'Extremal': 
    obj = ExtremalConstraints(v=params['v'], 
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
    handle_constraints('Volume', x, params, epi_constraint=True)


if __name__ == "__main__":
    main()
