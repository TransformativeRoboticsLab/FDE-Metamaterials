import sys
import os
import numpy as np
import pytest
from fenics import UnitSquareMesh

# Add the parent directory (which contains both 'test' and 'meta_design') to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_design.metamaterial import Metamaterial
from meta_design.filters import DensityFilter
from meta_design.main import PoissonRatio

def finite_difference_gradient(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (f(x_plus, np.array([])) - f(x_minus, np.array([]))) / (2 * eps)
        print(f"{i}/{len(x)}")
    return grad

@pytest.fixture
def setup_metamaterial():
    print("Setup metamaterial")
    nelx = 10  # Reduced size for faster tests
    nely = nelx
    E_max = 1.
    E_min = 1e-9
    nu = 0.3

    metamate = Metamaterial(E_max, E_min, nu)
    metamate.mesh = UnitSquareMesh(nelx, nely, 'crossed')
    metamate.create_function_spaces()

    filt = DensityFilter(metamate.mesh, 0.1, distance_method='periodic')

    return metamate, filt

def test_poisson_ratio_gradient(setup_metamaterial):
    metamate, filt = setup_metamaterial
    pr = PoissonRatio(metamaterial=metamate, filt=filt, beta=1, eta=0.5, plot=False, filter_and_project=False)

    # Initialize x with random values
    x = np.random.rand(metamate.R.dim())

    # Compute gradient using your method
    print("Computing fenics gradient")
    your_grad = np.zeros_like(x)
    pr(x, your_grad)

    print("Computing finite difference gradient")
    fd_grad = -finite_difference_gradient(pr, x)

    relative_error = np.linalg.norm(your_grad - fd_grad) / np.linalg.norm(fd_grad)
    print(f"Relative error: {relative_error}")

    assert relative_error < 1e-5, f"Relative error {relative_error} is too large"

if __name__ == "__main__":
    pytest.main([__file__])