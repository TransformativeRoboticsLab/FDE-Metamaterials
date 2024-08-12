from fenics import *
# from fenics_adjoint import *
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from metamaterial import Metamaterial
from filters import DensityFilter
from optimization import VolumeConstraint, IsotropicConstraint, BulkModulusConstraint, ShearModulusConstraint, OptimizationState, Epigraph, ExtremalConstraints, AndreassenOptimization, MaterialSymmetryConstraints

# we have the ability to pass in an objective function.
# we do this because on the back end only the objective actually solves the FEM problem, and then passes around the information to the constraints because none of that is going to change.
# basically we do not need to resolve the FEM problem for every constraint
# however, this doesn't work when we want to do a finite difference check
# this is because we pertube the input x every time, but the constraint actually just ignore that information anyway
# hence we have to call the objective function with the perturbed x, and then call the constraint (where the x doesn't matter)
def finite_difference_checker(func, x, grad_analytical, epsilon=1e-5, obj=None):
    grad_fd = np.zeros_like(grad_analytical)
    perturb = np.zeros_like(x)
    r_plus = np.zeros(func.n_constraints)
    r_minus = np.zeros(func.n_constraints)
    
    for i in tqdm(range(grad_fd.shape[1]), desc="Checking gradient"):
        perturb[i] = epsilon
        x_plus = x + perturb
        x_minus = x - perturb
        
        obj(np.zeros(obj.n_constraints), x_plus, np.array([])) if obj is not None else None
        func(r_plus, x_plus, np.array([]))
        
        obj(np.zeros(obj.n_constraints), x_minus, np.array([])) if obj is not None else None
        func(r_minus, x_minus, np.array([]))
        
        grad_fd[:,i] = (r_plus - r_minus) / (2. * epsilon)
        perturb[i] = 0

    diff = np.linalg.norm(grad_analytical - grad_fd)
    fig, ax = plt.subplots()
    ax.plot(grad_analytical.flatten(), label='Analytical')
    ax.plot(grad_fd.flatten(), label='Finite Difference')
    plt.legend()
    plt.show()
    rel_diff = diff / (np.linalg.norm(grad_analytical) + np.linalg.norm(grad_fd))

    # print(f"Finite Difference Gradient: {grad_fd}")
    # print(f"Analytical Gradient: {grad_analytical}")
    # print(f"Gradient difference: {grad_analytical - grad_fd}")
    print(f"Absolute difference: {diff:.2e}")
    print(f"Relative difference: {rel_diff:.2e}")

    return grad_fd, diff, rel_diff

def main():
    nelx = 11
    nely = nelx
    E_max = 1.
    E_min = 1e-9
    nu = 0.3
    vol_frac = 0.5
    start_beta, n_betas = 1, 8
    eta = 0.5
    pen = 3.
    a = 2e-3
    optim_type = 'pr'

    metamate = Metamaterial(E_max, E_min, nu)
    # metamate.mesh = UnitSquareMesh(nelx, nely, 'crossed')
    metamate.mesh = RectangleMesh.create([Point(0, 0), Point(1, 1)], [nelx, nely], CellType.Type.quadrilateral)
    metamate.create_function_spaces()
    
    filt = DensityFilter(metamate.mesh, 0.05, distance_method='periodic')
    
    ops = OptimizationState()
    ops.beta = start_beta
    ops.eta = eta
    ops.pen = pen
    ops.filt = filt

    dim = metamate.R.dim()
    np.random.seed(0)
    x = np.random.uniform(1e-3, 1, dim)
    print(f"Checking {dim:d} dimensional gradient\n")

    f = AndreassenOptimization(optim_type=optim_type, metamaterial=metamate, ops=ops, plot=False, verbose=False, filter_and_project=False)
    g_vol = VolumeConstraint(V=vol_frac, ops=ops, verbose=False)
    g_iso = IsotropicConstraint(eps=1e-5, ops=ops, verbose=False)
    g_blk = BulkModulusConstraint(E_max, nu, a=a, ops=ops, verbose=False)
    g_shr = ShearModulusConstraint(E_max, nu, a=a, ops=ops, verbose=False)

    grad_f = np.zeros_like(x)
    grad_g_vol = np.zeros_like(x)
    grad_g_iso = np.zeros_like(x)
    grad_g_blk = np.zeros_like(x)
    grad_g_shr = np.zeros_like(x)
    
    # compute gradients from the functions themselves
    # f(x, grad_f)
    # g_vol(x, grad_g_vol)
    # g_iso(x, grad_g_iso)
    # g_blk(x, grad_g_blk)
    # g_shr(x, grad_g_shr)
    
    # Epigraph forms
    epi = Epigraph()
    v = np.array([[0., 1., 0.],
                  [1., 0., 0.],
                  [0., 0., 1.]])
    g_ext = ExtremalConstraints(v=v, extremal_mode=2, metamaterial=metamate, ops=ops, verbose=False, plot=False, )
    g_sym = MaterialSymmetryConstraints(ops=ops, eps=1e-3, symmetry_order='rectangular', verbose=False)
    x = np.append(x, 1.)
    grad_epi = np.zeros_like(x)
    grad_extrem = np.zeros((g_ext.n_constraints, x.size))
    grad_g_sym = np.zeros((g_sym.n_constraints, x.size))

    # epi(x, grad_epi)
    # print("Checking Epigraph Gradient:")
    # finite_difference_checker(epi, x, grad_epi)

    g_ext(np.ones(g_ext.n_constraints), x, grad_extrem)
    # print("Checking Extremal Gradient:")
    # finite_difference_checker(g_ext, x, grad_extrem)
    
    g_sym(np.ones(g_sym.n_constraints), x, grad_g_sym)
    print("Checking Symmetry Gradient:")
    finite_difference_checker(g_sym, x, grad_g_sym, obj=g_ext)
    
    
    
    
    # Check gradients using finite differences
    # print("Checking Objective Gradient:")
    # finite_difference_checker(f, x, grad_f)
    
    # print("\nChecking Isotropic Constraint Gradient:")
    # finite_difference_checker(g_iso, x, grad_g_iso, obj=f)
    
    # print("\nChecking Bulk Modulus Constraint Gradient:")
    # finite_difference_checker(g_blk, x, grad_g_blk, obj=f)
    
    # print("\nChecking Shear Modulus Constraint Gradient:")
    # finite_difference_checker(g_shr, x, grad_g_shr, obj=f)
    
    # print("\nChecking Volume Constraint Gradient:")
    # finite_difference_checker(g_vol, x, grad_g_vol)
    
    

if __name__ == "__main__":
    main()
