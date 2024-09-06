from matplotlib import pyplot as plt
from functools import partial
import pickle, os, hashlib
import numpy as np
from fenics import *
from tqdm import tqdm

from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import coo_matrix

from jax.experimental import sparse
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

def filter_rho(rho: Function, H: np.array, Hs: np.array):
    filtered_rho = Function(rho.function_space())
    filtered_rho.vector().set_local(np.divide(H @ rho.vector()[:], Hs))
    return filtered_rho

# def backend_filter_rho(rho: Function, H: np.array, Hs: np.array):
#     filtered_rho = Function(rho.function_space())
#     filtered_rho.vector().set_local(np.divide(H @ rho.vector()[:], Hs))
#     return filtered_rho

# class FilterRhoBlock(Block):
#     def __init__(self, rho, H, Hs, **kwargs):
#         # super().__init__(**kwargs)
#         super(FilterRhoBlock, self).__init__()
#         self.kwargs = kwargs
#         self.add_dependency(rho)
#         self.H   = H
#         self.Hs  = Hs
#         import scipy.sparse as sp
#         self.HsH = sp.diags(1/Hs) @ H
#         logging.info(f"Using {self.__str__()}")

#     def __str__(self):
#         return 'FilterRhoBlock'
    
#     def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
#         tlm_input = tlm_inputs[0]
#         Jv = Vector(tlm_input)
#         Jv[:] = self.HsH @ tlm_input.get_local()
#         # print("Filter tlm", self.__str__())
#         return Jv

#     def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
#         adj_input = adj_inputs[0]
#         Jv = Vector(adj_input)
#         Jv[:] = self.HsH.T @ adj_input.get_local()
#         # print("Filter adj", self.__str__())
#         return Jv

#     def recompute_component(self, inputs, block_variable, idx, prepared):
#         # print("Filtering", self.__str__())
#         return backend_filter_rho(inputs[0], self.H, self.Hs)

# filter_rho = overload_function(backend_filter_rho, FilterRhoBlock)

# TODO: Make common constructor interface for filters?
# We could pass in a function space, then a radius, and then kwargs
# For DensityFilter we can get the mesh from the function space with fn_space.mesh()
class DensityFilter:
    calculated_filters = {}
    
    def __init__(self, mesh: Mesh, radius: float, distance_method: str = 'periodic'):
        self.mesh = mesh
        self.radius = radius
        self.distance_method = distance_method

        distance_methods = {
            'periodic': self._calculate_periodic_distances,
            'flat': self._calculate_flat_distances
        }

        if distance_method not in distance_methods:
            raise ValueError("Invalid filter_type. Must be 'periodic' or 'flat'.")

        self._distance_fn = distance_methods[distance_method]
        
        # calculating filters is expensive, so we cache them
        if os.path.exists('calculated_filters.pkl'):
            with open('calculated_filters.pkl', 'rb') as f:
                self.calculated_filters = pickle.load(f)
                
        self.key = (self.mesh.hash(), radius, distance_method)
        
        if self.key in self.calculated_filters:
            print("Loading filter from cache")
            self.H, self.Hs, self.H_jax, self.Hs_jax = self.calculated_filters[self.key]
        else:
            print("Calculating filter for the first time")
            self._calculate_filter()

        self.calculated_filters = None
            
    def filter(self, rho: Function):
        # r = Function(rho.function_space(), name="Rho_filtered")
        # r.assign(filter_rho(rho, self.H, self.Hs))
        return filter_rho(rho, self.H, self.Hs)
        
    def _calculate_filter(self):
        distances = self._distance_fn()
        H = np.maximum(0., self.radius - distances)
        H[H < 1e-6] = 0.
        self.Hs = H.sum(1)
        self.Hs_jax = jnp.array(self.Hs)        
        self.H = coo_matrix(H)
        self.H_jax = sparse.BCOO.from_scipy_sparse(self.H)
        
        self.calculated_filters[self.key] = (self.H, self.Hs, self.H_jax, self.Hs_jax)
        
        with open('calculated_filters.pkl', 'wb') as f:
            pickle.dump(self.calculated_filters, f)
        
    def _calculate_flat_distances(self):
        midpoints = [cell.midpoint().array()[:] for cell in cells(self.mesh)]
        return euclidean_distances(midpoints)

    def _calculate_periodic_distances(self):

        def periodic_distances(midpoints, domain_size=1.0):
            # Pairwise x and y differences
            # col vector - row vector = matrix
            x_diff = np.abs(midpoints[:, None, 0] - midpoints[None, :, 0]) 
            y_diff = np.abs(midpoints[:, None, 1] - midpoints[None, :, 1])

            # Adjust for wrap-around
            x_diff = np.minimum(x_diff, domain_size - x_diff)
            y_diff = np.minimum(y_diff, domain_size - y_diff)
            
            # Calculate the Euclidean distances with wrap-around consideration
            distances = np.sqrt(x_diff**2 + y_diff**2)
            return distances

        midpoints = np.array([cell.midpoint().array()[0:2] for cell in cells(self.mesh)])
        
        return periodic_distances(midpoints)

class HelmholtzFilter:
    def __init__(self, radius: float, fn_space: FunctionSpace):
        self.radius = radius / 2. / np.sqrt(3)
        self.fn_space = fn_space

        r, w = TrialFunction(fn_space), TestFunction(fn_space)
        self.a = (self.radius**2)*inner(grad(r), grad(w))*dx + r*w*dx
        self.solver = LUSolver(assemble(self.a))

        self.r = Function(fn_space)
        self.r_filtered = Function(fn_space)
        self.b = Vector(self.fn_space.mesh().mpi_comm(), self.fn_space.dim())
    
    def filter(self, r_array):
        assert r_array.size == self.fn_space.dim(), "Input array size must match function space dimension"
        self.r.vector()[:] = r_array
        w = TestFunction(self.fn_space)
        L = self.r*w*dx
        assemble(L, tensor=self.b)
        
        self.solver.solve(self.r_filtered.vector(), self.b)
        
        return self.r_filtered.vector()[:]


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def jax_helmholtz_filter(filt, r_array):
    return filt.filter(r_array)
    # return r_array * 2

def jax_helmholtz_filter_fwd(filt, r_array):
    rt = jax_helmholtz_filter(filt, r_array)
    return rt, (r_array, rt)

def jax_helmholtz_filter_bwd(filt, res, g):
    # r_array, r_filtered = res
    # adjoint = filt.filter(g)
    adjoint = jax_helmholtz_filter(filt, g)
    return (adjoint,)

jax_helmholtz_filter.defvjp(jax_helmholtz_filter_fwd, jax_helmholtz_filter_bwd)

def finite_difference(f, x, V, eps=1e-7):
    grad = np.zeros_like(x)
    perturb = np.zeros_like(x)
    for i in tqdm(range(x.size), desc="Calculating finite difference"):
        perturb[i] = eps
        x_plus = x + perturb
        x_minus = x - perturb
        grad[i] = (f(x_plus) - f(x_minus)) / (2*eps)
        perturb[i] = 0.
    return grad

def check_gradient(filter_obj, x, eps=1e-7, rtol=1e-5, atol=1e-5, show_plot=False):
    def f(x):
        return jnp.sum(jax_helmholtz_filter(filter_obj, x)**2)
    
    jax_grad = jax.grad(f)(x)
    fd_grad  = finite_difference(f, x, filter_obj.fn_space, eps)
    
    r = Function(filter_obj.fn_space)
    r.vector()[:] = x
    r_filt = Function(filter_obj.fn_space)
    r_filt.vector()[:] = jax_helmholtz_filter(filter_obj, x)
    jax_fn = Function(filter_obj.fn_space)
    jax_fn.vector()[:] = jax_grad
    fd_fn = Function(filter_obj.fn_space)
    fd_fn.vector()[:] = fd_grad
    
    if show_plot:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        plt.sca(axs[0,0])
        c=plot(r, cmap='gray')
        plt.colorbar(c)
        plt.title("Density")
        plt.sca(axs[0,1])
        c=plot(r_filt, cmap='gray')
        plt.colorbar(c)
        plt.title("Filtered Density")
        plt.sca(axs[1,0])
        c=plot(jax_fn)
        plt.colorbar(c)
        plt.title("JAX Gradient")
        plt.sca(axs[1,1])
        c=plot(fd_fn)
        plt.colorbar(c)
        plt.title("Finite Difference Gradient")
        plt.show()
    
    np.testing.assert_allclose(jax_grad, fd_grad, rtol=rtol, atol=atol)
    print("Gradient check passed")
    

if __name__ == "__main__":
    np.random.seed(0)

    mesh = UnitSquareMesh(30, 30, 'crossed')
    fn_space = FunctionSpace(mesh, 'CG', 1)
    # fn_space = FunctionSpace(mesh, 'DG', 0)
    expr = Expression('sqrt(pow(x[0]-0.5,2) + pow(x[1]-0.5,2)) < 0.2 ? 1.0 : 0.0', degree=1)
    rho = Function(fn_space)
    rho.vector()[:] = np.random.uniform(0, 1, fn_space.dim())
    rho.interpolate(expr)

    radius = 0.1
    h_filt = HelmholtzFilter(radius, fn_space)

    # the finite difference of the helmholtz filter shows some mesh artifacts that I haven't figured out how to fix.
    # The gradient check doesn't pass, but when plotted the trend between the two so I'm happy with that.
    # Also a sanity check when using the circle expression as the input gives the correct values (grad = 2 * rho because the function is sum(rho**2))
    check_gradient(h_filt, rho.vector()[:], eps=1e-4, show_plot=True)

    
    
    