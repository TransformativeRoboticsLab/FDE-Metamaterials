import inspect
from functools import partial

import fenics as fe
import jax
import matplotlib.pyplot as plt
import numpy as np
from filters import (DensityFilter, HelmholtzFilter, jax_density_filter,
                     jax_helmholtz_filter)
from metamaterial import Metamaterial
from optimization import (EigenvectorConstraint, EnergyObjective, Epigraph,
                          ExtremalConstraints, GeometricConstraints,
                          InvariantsConstraint, OptimizationState)
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

ISQR2 = 1. / np.sqrt(2.)
v_dict = {
    "BULK": np.array([[ISQR2, -ISQR2, 0.],
                      [ISQR2,  ISQR2, 0.],
                      [0.,     0.,    1.]]),
    "IBULK": np.array([[-ISQR2, ISQR2, 0.],
                       [ISQR2, ISQR2, 0.],
                       [0.,    0.,    1.]]),
    "VERT": np.array([[0., 1., 0.],
                      [1., 0., 0.],
                      [0., 0., 1.]]),
    "VERT2": np.array([[0., ISQR2, -ISQR2],
                       [1., 0.,     0.],
                       [0., ISQR2,  ISQR2]]),
    "SHEAR": np.array([[0., -ISQR2, ISQR2],
                       [0.,  ISQR2, ISQR2],
                       [1.,  0.,    0.]]),
    "SHEARXY": np.array([[0., 1., 0.],
                         [0., 0., 1.],
                         [1., 0., 0.]]),
    "HSA": np.array([[0.,     0.,    1.],
                     [ISQR2, -ISQR2, 0.],
                     [ISQR2,  ISQR2, 0.]]),
    "HSA2": np.array([[0.,     0.,    1.],
                      [ISQR2,  ISQR2, 0.],
                      [-ISQR2, ISQR2, 0.]]),
    "IHSA": np.array([[1., 0., 0.],
                      [0., ISQR2, -ISQR2],
                      [0., ISQR2,  ISQR2]]),
    "EYE": np.eye(3),
}


def finite_difference_checker(constraint, x, grad_analytical, params, epsilon=1e-5):
    args_count = len(inspect.signature(constraint).parameters)
    grad_fd = np.zeros_like(grad_analytical)

    for i in tqdm(range(grad_fd.shape[grad_fd.ndim-1]), desc="Checking gradient"):
        perturb = np.zeros_like(x)
        perturb[i] = epsilon
        x_plus = x + perturb
        x_minus = x - perturb


        if args_count > 2:  # fn(results, x, grad)
            r_plus = np.zeros(constraint.n_constraints)
            r_minus = np.zeros(constraint.n_constraints)
            if params['obj']:
                params['obj'](np.zeros(params['obj'].n_constraints), 
                              x_plus, np.array([]))
            constraint(r_plus, x_plus, np.array([]))
            if params['obj']:
                params['obj'](np.zeros(params['obj'].n_constraints), 
                              x_minus, np.array([]))
            constraint(r_minus, x_minus, np.array([]))
            grad_fd[:, i] = (r_plus - r_minus) / (2 * epsilon)
        elif args_count == 2:  # c = fn(x, grad)
            c_plus = constraint(x_plus, np.array([]))
            c_minus = constraint(x_minus, np.array([]))
            grad_fd[i] = (c_plus - c_minus) / (2 * epsilon)

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
                                        plot=params['plot']),
        'Energy': EnergyObjective(v=params['v'], 
                                   extremal_mode=params['extremal_mode'], 
                                   metamaterial=params['metamaterial'], ops=params['ops'], 
                                   verbose=params['verbose'], 
                                   plot=params['plot']),
        'Geometric': GeometricConstraints(ops=params['ops'], 
                                          metamaterial=params['metamaterial'],
                                          line_width=params['line_width'], 
                                          line_space=params['line_space'], 
                                          c=1./params['metamaterial'].mesh.hmin(),
                                          eps=1.,
                                          verbose=params['verbose']),
        'Invariants': InvariantsConstraint(ops=params['ops'], 
                                           verbose=params['verbose']),
        'Eigenvector': EigenvectorConstraint(v=params['v'],
                                             ops=params['ops'],
                                             verbose=params['verbose']),
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
        'v': v_dict[v_basis],
        'extremal_mode': 1,
        'verbose': False,
        'plot': False,
        'obj': None,  
        'line_width': norm_line_width,
        'line_space': norm_line_space,
    }

    metamate = params['metamaterial']
    metamate.create_function_spaces()
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
                        plot=params['plot'])
    params['obj'] = obj
    
    # handle_constraints('Invariants', x, params, epi_constraint=True)
    handle_constraints('Eigenvector', x, params, epi_constraint=True)
    # handle_constraints('Geometric', x, params, epi_constraint=True)


if __name__ == "__main__":
    main()
