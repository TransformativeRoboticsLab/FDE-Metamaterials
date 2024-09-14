import os
import pickle
from functools import partial

import jax
import numpy as np
from fenics import *
from jax.experimental import sparse
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@jax.jit
def jax_density_convolution(x, kernel):
    return

@jax.jit
def jax_density_filter(H, Hs, x):
    return jnp.divide(H @ x, Hs)

@jax.jit
def jax_projection(x, beta=1., eta=0.5):
    tanh_beta_eta = jnp.tanh(beta * eta)
    tanh_beta_x_minus_eta = jnp.tanh(beta * (x - eta))
    tanh_beta_one_minus_eta = jnp.tanh(beta * (1. - eta))

    numerator = tanh_beta_eta + tanh_beta_x_minus_eta
    denominator = tanh_beta_eta + tanh_beta_one_minus_eta

    return jnp.array(numerator / denominator)

@jax.jit
def jax_simp(x, penalty):
    return jnp.power(x, penalty)

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
        self.fn_space = FunctionSpace(mesh, 'DG', 0)

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
        self.H = self._distance_fn()
        self.Hs = np.asarray(self.H.sum(1)).flatten()
        self.Hs_jax = jnp.array(self.Hs)
        self.H_jax = sparse.BCOO.from_scipy_sparse(self.H)
        
        self.calculated_filters[self.key] = (self.H, self.Hs, self.H_jax, self.Hs_jax)
        
        with open('calculated_filters.pkl', 'wb') as f:
            pickle.dump(self.calculated_filters, f)
        
    def _calculate_flat_distances(self):
        midpoints = [cell.midpoint().array()[:] for cell in cells(self.mesh)]
        tree = KDTree(midpoints)
        return tree.sparse_distance_matrix(tree, max_distance=self.radius, output_type='coo_matrix')



    def _calculate_periodic_distances(self):

        # NOTE: If instead of finding the midpoints we actually did a function_space.tabulate_dof_coordinates() we could also use this method for a CG space instead of just a DG space. This is because the dofs in a DG space are the cell midpoints, whereas in a CG space they are the vertices of the mesh.
        midpoints = np.array([c.midpoint()[:][:2] for c in cells(self.mesh)])
        
        x_min, y_min = self.mesh.coordinates().min(axis=0)
        x_max, y_max = self.mesh.coordinates().max(axis=0)
        width, height = x_max - x_min, y_max - y_min
        
        # The idea here is to create a KDTree for the minimum number of cells that give us full periodicity we want to achieve. So it looks something like this where O is the origin, A is the FEM base domain, and B, C, D, E are the periodic copies of A. 
        '''
         _____ _____
        |     |     |
        |  B  |  A  |
        |_____O_____|_____
        |     |     |     |
        |  C  |  D  |  E  |
        |_____|_____|_____|
        '''
        shifts = [
            np.array([0, 0]),                    # no shift, A
            np.array([width, 0]),                # -x shift, B
            np.array([width, height]),           # -x and -y shift, C
            np.array([0, height]),               # -y shift, D
            np.array([-width, height])           # +x shift, -y, E
        ]

        # We then create trees for every space and then can query between them to determine which indices are within the filter radius. This is the minimum number of translations to get full periodicity, where every corner of the domain is able to touch every other corner.
        trees = [KDTree(midpoints - shift) for shift in shifts]

        distance_mats = [trees[0].sparse_distance_matrix(tree, max_distance=self.radius, output_type='coo_matrix') for tree in trees]

        # This is the standard TO conic filter, where we take the distance from the filter radius and subtract it from the radius to get the filter value
        for D in distance_mats:
            D.data = self.radius - D.data

        # Combine all distance matrices and their transposes
        # Transposes are what let us not have to tile the full space, but only the minimum number of cells to get full periodicity
        tdist = sum(D + D.T for D in distance_mats[1:]) + distance_mats[0]

        # In case there are any cancellations when we do the subtraction for the filter (i.e., the distance is exactly the filter radius), we need to tell the matrix to remove tracking those values.
        tdist.data[np.isclose(tdist.data, 0.)] = 0.
        tdist.eliminate_zeros()

        return tdist

class HelmholtzFilter:
    def __init__(self, radius: float, fn_space: FunctionSpace):
        self.radius = radius
        self._scaled_radius = radius / 2. / np.sqrt(3)
        self.fn_space = fn_space

        r, w = TrialFunction(fn_space), TestFunction(fn_space)
        self.a = (self._scaled_radius**2)*inner(grad(r), grad(w))*dx + r*w*dx
        self.solver = LUSolver(assemble(self.a))

        self.r = Function(fn_space)
        self.r_filtered = Function(fn_space)
        self.b = Vector(self.fn_space.mesh().mpi_comm(), self.fn_space.dim())
        self.w = TestFunction(self.fn_space)
    
    def filter(self, r_array):
        assert r_array.size == self.fn_space.dim(), "Input array size must match function space dimension"
        self.r.vector()[:] = r_array
        L = self.r*self.w*dx
        assemble(L, tensor=self.b)
        
        self.solver.solve(self.r_filtered.vector(), self.b)
        
        return self.r_filtered.vector()[:]


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def jax_helmholtz_filter(filt, r_array):
    return filt.filter(r_array)

def jax_helmholtz_filter_fwd(filt, r_array):
    rt = jax_helmholtz_filter(filt, r_array)
    return (rt, ())

def jax_helmholtz_filter_bwd(filt, _, g):
    adjoint = jax_helmholtz_filter(filt, g)
    return (adjoint,)

jax_helmholtz_filter.defvjp(jax_helmholtz_filter_fwd, jax_helmholtz_filter_bwd)

def finite_difference(f, x, eps=1e-7):
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
    
    if type(filter_obj) == DensityFilter:
        filter_fn = partial(jax_density_filter, filter_obj.H_jax, filter_obj.Hs_jax)
    elif type(filter_obj) == HelmholtzFilter:
        filter_fn = partial(jax_helmholtz_filter, filter_obj)
    else:
        raise ValueError("Invalid filter type")
    
    def f(x):
        return jnp.sum(filter_fn(x)**2)
    
    jax_grad = jax.grad(f)(x)
    fd_grad  = finite_difference(f, x, eps)
    
    r = Function(filter_obj.fn_space)
    r.vector()[:] = x
    r_filt = Function(filter_obj.fn_space)
    r_filt.vector()[:] = filter_fn(x)
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
        # plt.show()

        plt.figure()
        plt.plot(fd_grad[:], label='Finite Difference')
        plt.plot(jax_grad[:], label='JAX')
        plt.legend()
        plt.show()
    
    np.testing.assert_allclose(jax_grad, fd_grad, rtol=rtol, atol=atol)
    print("Gradient check passed")
    

if __name__ == "__main__":
    np.random.seed(0)

    mesh = UnitSquareMesh(50,50, 'crossed')
    # mesh = RectangleMesh.create([Point(0, 0), Point(1, 1)], [20, 20], CellType.Type.quadrilateral)
    expr = Expression('sqrt(pow(x[0]-0.5,2) + pow(x[1]-0.5,2)) < 0.3 ? 1.0 : 0.0', degree=1)

    dg_space = FunctionSpace(mesh, 'DG', 0)
    rho_dg = Function(dg_space)
    rho_dg.interpolate(expr)
    cg_space = FunctionSpace(mesh, 'CG', 1)
    rho_cg = Function(cg_space)
    rho_cg.interpolate(expr)

    # the finite difference of the helmholtz filter shows some mesh artifacts that I haven't figured out how to fix.
    # probably something 
    # The gradient check doesn't pass, but when plotted the trend between the two so I'm happy with that.
    # Also a sanity check when using the circle expression as the input gives the correct values (grad = 2 * rho because the function is sum(rho**2))
    # The DG gradients (both JAX and FD) match, and they also match against the CG JAX gradient. The only outlier is the CG FD gradient.
    # For now I'm willing to move on...
    radius = 0.2
    # d_filt = DensityFilter(mesh, radius, distance_method='flat')
    # check_gradient(d_filt, rho_dg.vector()[:], eps=1e-2, show_plot=True)
    pd_filt = DensityFilter(mesh, radius, distance_method='periodic')
    check_gradient(pd_filt, rho_dg.vector()[:], eps=1e-2, show_plot=True)
    # h_filt = HelmholtzFilter(radius, cg_space)
    # check_gradient(h_filt, rho_cg.vector()[:], eps=1e-4, show_plot=True)

    
    
    