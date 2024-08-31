from optimization import OptimizationState, Epigraph, ExtremalConstraints, EnergyConstraint, GeometricConstraints, InvariantsConstraint
from filters import DensityFilter
from metamaterial import Metamaterial
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import inspect

import fenics as fe
import jax
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

        if params['obj']:
            params['obj'](np.zeros(params['obj'].n_constraints),
                          x_plus, np.array([]))
            params['obj'](np.zeros(params['obj'].n_constraints),
                          x_minus, np.array([]))

        if args_count > 2:  # fn(results, x, grad)
            r_plus = np.zeros(constraint.n_constraints)
            r_minus = np.zeros(constraint.n_constraints)
            constraint(r_plus, x_plus, np.array([]))
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
    ax[1].plot(np.abs(grad_analytical - grad_fd).flatten() /
               (grad_analytical.flatten()+1e-8), label='Relative Difference')
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
        'Energy': EnergyConstraint(v=params['v'], 
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
                                           verbose=params['verbose'])
    }

    # if the constraint is formulated for an epigraph form we need to add a DOF for the t variable
    if epi_constraint:
        x = np.append(x, 0.)

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
    nelx = 11
    nely = nelx
    E_max, E_min, nu = 1., 1e-9, 0.3
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
    mesh_type = 'quad'
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

    params['metamaterial'].create_function_spaces()
    params['ops'].filt = DensityFilter(params['metamaterial'].mesh,
                                       norm_filter_radius, distance_method='periodic')
    # Initial design variables
    # x = np.random.uniform(1e-3, 1, params['metamaterial'].R.dim())
    x = np.ones((nely, nelx))
    x[0:3,0:3] = 0.
    x[3:5,3:5] = 0.25
    x[5:7,5:7] = 0.5
    x[7:11,7:11] = 0.75
    x[-1,-1] = 1.
    x = x.flatten()

    # handle_constraints('Epigraph', x, params, epi_constraint=True)
    # handle_constraints('Extremal', x, params, epi_constraint=True)
    # handle_constraints('Energy', x, params)
    # handle_constraints('Geometric', x, params, epi_constraint=True)
    handle_constraints('Invariants', x, params)


if __name__ == "__main__":
    main()
