import numpy as np
from tqdm import tqdm
from metamaterial import Metamaterial
from filters import DensityFilter
from optimization import  OptimizationState, VolumeConstraint
from fenics import *

def finite_difference_checker(func, x, grad_analytical, eps=1e-3):
    grad_fd = np.zeros_like(x)
    perturb = np.zeros_like(x)
    
    for i in tqdm(range(len(x)), desc="Checking gradients"):
        perturb[i] = eps
        f_plus = func(x + perturb, np.zeros_like(x))
        f_minus = func(x - perturb, np.zeros_like(x))
        grad_fd[i] = (f_plus - f_minus) / (2 * eps)
        perturb[i] = 0

    diff = np.linalg.norm(grad_analytical - grad_fd)
    rel_diff = diff / (np.linalg.norm(grad_analytical) + np.linalg.norm(grad_fd))

    # print(f"Finite Difference Gradient: {grad_fd}")
    # print(f"Analytical Gradient: {grad_analytical}")
    print(f"Absolute difference: {diff}")
    print(f"Relative difference: {rel_diff}")

    return grad_fd, diff, rel_diff

# Example usage with VolumeConstraint
def main():
    nelx = 2
    nely = nelx
    E_max = 1.
    E_min = 1e-9
    nu = 0.3
    vol_frac = 0.35
    start_beta, n_betas = 8, 8
    betas = [start_beta * 2 ** i for i in range(n_betas)]
    # print(betas)
    eta = 0.5
    pen = 3.
    epoch_duration = 50
    a = 2e-3
    optim_type = 'shear'

    metamate = Metamaterial(E_max, E_min, nu)
    metamate.mesh = UnitSquareMesh(nelx, nely, 'crossed')
    metamate.create_function_spaces()
    # metamate.plot_mesh()
    
    filt = DensityFilter(metamate.mesh, 0.05, distance_method='periodic')
    
    ops = OptimizationState()
    ops.beta = start_beta
    ops.eta = eta
    ops.pen = pen
    ops.filt = filt
    dim = metamate.R.dim()
    # Initialize the design variable
    np.random.seed(0)
    x = np.random.uniform(0,1,dim).astype(np.float64)  # Example design variable, modify as needed
    print(f"Metamaterial Cell Volume: {metamate.cell_vol:3f} for {dim:d} cells")

    g_vol = VolumeConstraint(V=vol_frac, ops=ops, verbose=True)
    # Run the gradient checker for the volume constraint
    check_volume_constraint_gradient(g_vol, x)

if __name__ == "__main__":
    main()
