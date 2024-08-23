import jax.interpreters
import matplotlib.pyplot as plt
from matplotlib import gridspec
from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np
import jax
from jax.experimental import sparse
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from fenics import *

from filters import DensityFilter
from image import bitmapify

@jax.jit
def jax_density_filter(x, H, Hs):
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

class Epigraph:
    """
    A class representing a minimax optimization problem.
    We reformulate the problem as a nonlinear optimization problem by adding a slack variable t as the objective function.
    
    min_x max{f1, f2, f3}
    s.t. g(x) <= 0
    
    becomes
    
    min_{x, t} t
    s.t. f1 <= t
         f2 <= t
         f3 <= t
         g(x) <= 0
         
    The objective function is simply the slack variable t.
    The gradiant of the objective function is then all zeros with a 1 at the end of the vector.
    """
    def __call__(self, x, grad):
        """
        Evaluates the objective function of the minimax problem.

        Parameters:
        - x: The input vector.
        - grad: The gradient vector.

        Returns:
        - The objective function value.
        """
        t = x[-1]
        
        if grad.size > 0:
            grad[:-1], grad[-1] = 0., 1.
        
        return t

    
    
class EnergyConstraint:
    """ Not in epigraph form"""
    def __init__(self, v, extremal_mode, metamaterial, ops, verbose = True, plot_interval = 10, plot = True):
        self.v = v
        self.extremal_mode = extremal_mode
        self.metamaterial = metamaterial
        self.ops = ops
        self.verbose = verbose
        self.plot_interval = plot_interval
        self.evals = []

        if plot:
            plt.ion()
            self.fig = plt.figure(figsize=(16,8))
            grid_spec = gridspec.GridSpec(2, 3, )
            self.ax1 = [plt.subplot(grid_spec[0, 0]), plt.subplot(grid_spec[0, 1]), plt.subplot(grid_spec[0, 2])]
            self.ax2 = plt.subplot(grid_spec[1, :])
        else:
            self.fig = None

    def __call__(self, x, grad):

        filt, beta, eta = self.ops.filt, self.ops.beta, self.ops.eta
        
        def filter_and_project(x):
            x = jax_density_filter(x, filt.H_jax, filt.Hs_jax)
            x = jax_projection(x, beta, eta)
            return x
        
        x_fem, dxfem_dx_vjp = jax.vjp(filter_and_project, x)
        
        self.metamaterial.x.vector()[:] = x_fem
        sols, Chom, _ = self.metamaterial.solve()
        E_max, nu = self.metamaterial.prop.E_max, self.metamaterial.prop.nu
        dChom_dxfem = self.metamaterial.homogenized_C(sols, E_max, nu)[1]
        
        self.ops.update_state(sols, Chom, dChom_dxfem, dxfem_dx_vjp, x_fem)
        
        def obj(C):
            m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
            C = m @ C @ m
            S = jnp.linalg.inv(C)
            
            if self.extremal_mode == 2:
                C, S = S, C
                
            vCv = self.v.T @ C @ self.v
            c1, c2, c3 = vCv[0,0], vCv[1,1], vCv[2,2]
            return c1**2/c2/c3
        
        c, dc_dChom = jax.value_and_grad(obj)(jnp.asarray(Chom))

        self.evals.append(c)
        
        if grad.size > 0:
            grad[:] = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]
            
            
        if self.verbose:
            print("-" * 30)
            print(f"Epoch {self.ops.epoch:d}, Step {len(self.evals):d}, Beta = {self.ops.beta:.1f}, Eta = {self.ops.eta:.1f}")
            print("-" * 30)
            print(f"Energy: {c:.4f}")
        
        if (len(self.evals) % self.plot_interval == 1) and self.fig is not None:
            self.update_plot(x)
            
        return float(c)

    def update_plot(self, x):
        filt, beta, eta = self.ops.filt, self.ops.beta, self.ops.eta
        x_tilde = jax_density_filter(x, filt.H_jax, filt.Hs_jax)
        x_bar   = jax_projection(x_tilde, beta, eta)
        fields = {f'x (V={np.mean(x):.3f})': x,
                    f'x_tilde (V={np.mean(x_tilde):.3f})': x_tilde,
                    f'x_bar beta={beta:d} (V={np.mean(x_bar):.3f})': x_bar}

        if len(fields) != len(self.ax1):
            raise ValueError("Number of fields must match number of axes")
        
        r = Function(self.metamaterial.R)
        for ax, (name, field) in zip(self.ax1, fields.items()):
            if field.size == self.metamaterial.R.dim():
                r.vector()[:] = field
                self.plot_density(r, title=f"{name}", ax=ax)
            else:
                raise ValueError("Field size does not match function space")
            ax.set_xticks([])
            ax.set_yticks([])
            
        self.ax2.clear()
        f_arr = np.asarray(self.evals)
        self.ax2.plot(range(1, len(self.evals)+1), f_arr, marker='o')  
        self.ax2.grid(True)
        self.ax2.set_xlim(left=0, right=len(self.evals) + 2) 
        if np.min(f_arr) > 0:
            self.ax2.set_yscale('log')
        else:
            self.ax2.set_ylim(-2, 2)
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(1e-3)
            
    def plot_density(self, r_in, title=None, ax=None):
        r = Function(r_in.function_space())
        r.vector()[:] = 1. - r_in.vector()[:]
        r.set_allow_extrapolation(True)
        
        if isinstance(ax, plt.Axes):
            plt.sca(ax)
        else:
            fig, ax = plt.subplots()
            
        ax.margins(x=0,y=0)

        # quad meshes aren't supported using the standard plot interface but we can convert them to an image and use imshow
        # the ordering of a quad mesh is row-major and imshow expects row-major so it works out
        cell_type = r_in.function_space().ufl_cell().cellname()
        if cell_type == 'quadrilateral':
            r_vec = r.vector()[:]
            # assume square space
            nely = np.sqrt(r_vec.size).astype(int)
            nelx = nely
            plt.imshow(r_vec.reshape((nely, nelx)), cmap='gray', vmin=0, vmax=1)
            ax.set_title(title)
            return
        
        plot(r, cmap='gray', vmin=0, vmax=1, title=title)
        
            

class ExtremalConstraints:
    """
    A class representing the original objective functions of the minimax problem.
    """
    
    def __init__(self, v, extremal_mode, metamaterial, ops, verbose=True, plot_interval = 10, plot=True):
        self.v = v
        self.extremal_mode = extremal_mode
        self.metamaterial = metamaterial
        self.ops = ops
        self.verbose = verbose
        self.plot_interval = plot_interval
        self.evals = []
                
        self.n_constraints = 4
        self.eps = 1.

        if plot:
            plt.ion()
            self.fig = plt.figure(figsize=(24, 8))
            grid_spec = gridspec.GridSpec(2, 6, )
            self.ax1 = [plt.subplot(grid_spec[0, 0]), plt.subplot(grid_spec[0, 1]), plt.subplot(grid_spec[0, 2]), plt.subplot(grid_spec[0, 3]), plt.subplot(grid_spec[0, 4]), plt.subplot(grid_spec[0, 5])]
            self.ax2 = plt.subplot(grid_spec[1, :])
        else:
            self.fig = None
        
        print(f"""
MinimaxConstraint initialized with:
v:
{v}
extremal_mode: {self.extremal_mode}
starting beta: {self.ops.beta}
verbose: {self.verbose}
plot_delay: {self.plot_interval}
""")

    def __call__(self, results, x, grad, dummy_run=False):
        
        x, t = x[:-1], x[-1]
        
        filt, beta, eta = self.ops.filt, self.ops.beta, self.ops.eta
        
        def filter_and_project(x):
            x = jax_density_filter(x, filt.H_jax, filt.Hs_jax)
            x = jax_projection(x, beta, eta)
            # x = jax_simp(x, 3.)
            return x
        
        x_fem, dxfem_dx_vjp = jax.vjp(filter_and_project, x)
        
        self.metamaterial.x.vector()[:] = x_fem
        sols, Chom, _ = self.metamaterial.solve()
        E_max, nu = self.metamaterial.prop.E_max, self.metamaterial.prop.nu
        dChom_dxfem = self.metamaterial.homogenized_C(sols, E_max, nu)[1]
        
        self.ops.update_state(sols, Chom, dChom_dxfem, dxfem_dx_vjp, x_fem)
            
        # def obj(C):
        #     m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
        #     C = self.v @ m @ C @ m @ self.v.T
        #     C /= jnp.linalg.norm(C, ord=2)
        #     w = jnp.linalg.eigvalsh(C)
        #     return jnp.array([w[0], 1.-w[1], 1.-w[2]])
        
        def obj(C):
            m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
            C = m @ C @ m
            C /= jnp.linalg.norm(C, ord=2)
            # print(np.linalg.eigvalsh(C))
            S = jnp.linalg.inv(C)
            S /= jnp.linalg.norm(S, ord=2)
            # print(np.linalg.eigvalsh(S))
            
            if self.extremal_mode == 2:
                C, S = S, C
            
            vCv = self.v.T @ C @ self.v
            c1, c2, c3 = vCv[0,0], vCv[1,1], vCv[2,2]
            # off_diag = jnp.linalg.norm(vCv - jnp.diag(jnp.diag(vCv)))
            # vSv = self.v.T @ S @ self.v
            # s1, s2, s3 = vSv[0,0], vSv[1,1], vSv[2,2]
            # print(c1, s2, s3)
            # return jnp.array([c1**2 / c2 / c3, off_diag])
            return jnp.array([c1, (1-c2), (1-c3), jnp.sum((vCv - jnp.diag(jnp.diag(vCv)))**2)])
        
        # Poisson's ratio test
        # def obj(C):
        #     S = jnp.linalg.inv(C)
        #     nu1 = -S[0][1] / S[1][1]
        #     nu2 = -S[1][0] / S[0][0]
        #     return jnp.array([nu1, nu2])
        
        # nullspace obj
        # this is really good at setting the eigenvectors for bimode isotropic
        # didn't do so well for unimode isotropic
        # def obj(C):
        #     from jax.numpy.linalg import norm
        #     m = jnp.diag(jnp.array([1., 1., np.sqrt(2)]))
        #     C = m @ C @ m
        #     S = jnp.linalg.inv(C)
        #     if self.extremal_mode == 2:
        #         C, S = S, C
        #     Cv = (C / jnp.linalg.norm(C, ord='fro')) @ self.v
        #     Sv = (S / jnp.linalg.norm(S, ord='fro')) @ self.v
        #     r1, r2, r3 = Cv[:,0], Sv[:,1], Sv[:,2]
        #     return jnp.array([norm(r1), norm(r2), norm(r3)])
        
        c = np.asarray(obj(jnp.asarray(Chom)))
        dc_dChom = jax.jacrev(obj)(jnp.asarray(Chom)).reshape((self.n_constraints,9))
        
        if grad.size > 0:
            for n in range(self.n_constraints):
                grad[n,:-1] = dxfem_dx_vjp(dc_dChom[n,:] @ dChom_dxfem)[0]
                grad[n,-1] = -1.
                
        results[:] = c - t
        
        if dummy_run:
            return
        
        self.evals.append([t, *c])
        if self.verbose:
            print("-" * 30)
            print(f"Epoch {self.ops.epoch:d}, Step {len(self.evals):d}, Beta = {self.ops.beta:.1f}, Eta = {self.ops.eta:.1f}")
            print("-" * 30)
            # print(f"g(x) = {c:.4f}")
            # print(t, c)
            print(f"t: {t:.3f} g(x): {c}")
        
        if (len(self.evals) % self.plot_interval == 1) and self.fig is not None:
            x_tilde = jax_density_filter(x, filt.H_jax, filt.Hs_jax)
            x_bar   = jax_projection(x_tilde, beta, eta)
            img_resolution = 200
            r_img = self.metamaterial.x.copy(deepcopy=True)
            x_img = np.flip(bitmapify(r_img, (1., 1.), (img_resolution, img_resolution)), axis=0)

            fields = {f'x (V={np.mean(x):.3f})': x,
                      f'x_tilde (V={np.mean(x_tilde):.3f})': x_tilde,
                      f'x_bar beta={beta:d} (V={np.mean(x_bar):.3f})': x_bar,
                      f'x_fem (V={np.mean(x_fem):.3f})': x_fem,
                      f'x_img': x_img,
                      f'Tiling': np.tile(x_img, (3,3))}
            self.update_plot(fields)

    def update_plot(self, fields):
        if len(fields) != len(self.ax1):
            raise ValueError("Number of fields must match number of axes")
        
        r = Function(self.metamaterial.R)
        for ax, (name, field) in zip(self.ax1, fields.items()):
            if field.size == self.metamaterial.R.dim():
                r.vector()[:] = field
                self.plot_density(r, title=f"{name}", ax=ax)
            else:
                ax.imshow(255 - field, cmap='gray')
                ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
            
        self.ax2.clear()
        f_arr = np.asarray(self.evals)
        self.ax2.plot(range(1, len(self.evals)+1), f_arr, marker='o')  
        self.ax2.grid(True)
        self.ax2.set_xlim(left=0, right=len(self.evals) + 2) 
        if np.min(f_arr) > 0:
            self.ax2.set_yscale('log')
        else:
            self.ax2.set_ylim(-2, 2)
            
        for iter_val in self.ops.epoch_iter_tracker:
            self.ax2.axvline(x=iter_val, color='black', linestyle='--', alpha=0.5, linewidth=3.)
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(1e-3)
            
    def plot_density(self, r_in, title=None, ax=None):
        r = Function(r_in.function_space())
        r.vector()[:] = 1. - r_in.vector()[:]
        r.set_allow_extrapolation(True)
        
        if isinstance(ax, plt.Axes):
            plt.sca(ax)
        else:
            fig, ax = plt.subplots()
            
        ax.margins(x=0,y=0)

        # quad meshes aren't supported using the standard plot interface but we can convert them to an image and use imshow
        # the ordering of a quad mesh is row-major and imshow expects row-major so it works out
        cell_type = r_in.function_space().ufl_cell().cellname()
        if cell_type == 'quadrilateral':
            r_vec = r.vector()[:]
            # assume square space
            nely = np.sqrt(r_vec.size).astype(int)
            nelx = nely
            plt.imshow(r_vec.reshape((nely, nelx)), cmap='gray', vmin=0, vmax=1)
            ax.set_title(title)
            return
        
        plot(r, cmap='gray', vmin=0, vmax=1, title=title)
        

class AndreassenOptimization:
    def __init__(self, optim_type, metamaterial, ops, verbose=True, plot=True, filter_and_project=True):
        self.optim_type = optim_type
        self.metamaterial = metamaterial
        self.ops = ops
        self.plot = plot
        self.verbose = verbose
        self.plot_interval = 20

        
        self.epoch = 0
        self.evals = []
        self.epoch_iter_tracker = []
        self.filter_and_project = filter_and_project
        
        
        if self.plot:
            plt.ion()
            self.fig = plt.figure(figsize=(16,8))
            grid_spec = gridspec.GridSpec(2, 4, )
            self.ax1 = [plt.subplot(grid_spec[0, 0]), plt.subplot(grid_spec[0, 1]), plt.subplot(grid_spec[0, 2]), plt.subplot(grid_spec[0, 3])]
            self.ax2 = plt.subplot(grid_spec[1, :])
            
        
    def __call__(self, x, grad):
        
        filt = self.ops.filt
        pen  = self.ops.pen
        beta = self.ops.beta
        eta  = self.ops.eta
        
        def filter_and_project(x):
            if not self.filter_and_project:
                return x
            x = jax_density_filter(x, filt.H_jax, filt.Hs_jax)
            x = jax_projection(x, beta, eta)
            x = jax_simp(x, pen)
            return x
        
        x_fem, dxfem_dx_vjp = jax.vjp(filter_and_project, x)
                
        self.metamaterial.x.vector()[:] = x_fem
        
        sols, Chom, uChom = self.metamaterial.solve()
        
        E_max = self.metamaterial.prop.E_max
        nu    = self.metamaterial.prop.nu
        # dChom_dxfem is a 3x3 list of lists that are fenics expressions. In a sense this a 3x3xN matrix, where the derivatives are with respect to the Nth component of the design vector.
        # This is achieved by applying the displacement solutions to a material of constant stiffness (E_max).
        # Xia has a good description in their paper, eq (22), but this is only for the derivative of the FEM solve. We handle all the other parts of the chain rule with JAX.
        dChom_dxfem = self.metamaterial.homogenized_C(sols, E_max, nu)[1]
        
        self.ops.update_state(sols, Chom, dChom_dxfem, dxfem_dx_vjp, x_fem)
        
        if self.optim_type == 'bulk':
            def obj(C):
                S = jnp.linalg.inv(C)
                return -1. / (S[0][0] + S[0][1]) / 2.
        elif self.optim_type == 'shear':
            obj = lambda C: -C[2][2]
        elif 'pr' in self.optim_type:
            def obj(C):
                S = jnp.linalg.inv(C)
                S = -S if self.optim_type == 'ppr' else S
                return -S[0][1]/S[0][0]
        else:
            raise ValueError("Invalid objective type")
            
        c, dc_dChom = jax.value_and_grad(obj)(jnp.asarray(Chom))
        
        self.evals.append(c)

        
        if grad.size > 0:
            grad[:] = dxfem_dx_vjp(np.asarray(dc_dChom).flatten() @ dChom_dxfem)[0]
            
        if (len(self.evals) % self.plot_interval == 1) and self.plot:
            x_tilde = jax_density_filter(x, filt.H_jax, filt.Hs_jax)
            x_bar   = jax_projection(x_tilde, beta, eta)
            fields = {f'x (V={np.mean(x):.3f})': x,
                      f'x_tilde (V={np.mean(x_tilde):.3f})': x_tilde,
                      f'x_bar beta={beta:d} (V={np.mean(x_bar):.3f})': x_bar,
                      f'x_fem (V={np.mean(x_fem):.3f})': x_fem}
            self.update_plot(fields)
            
        if self.verbose == True:
            print("-" * 30)
            print(f"Epoch {self.epoch}, Step {len(self.evals)}, Beta = {self.ops.beta}")
            print("-" * 30)
            print(f"f(x) = {c:.4f}")
            print(f"Constraints:")

        return float(c)
    
    def update_plot(self, fields):
        if len(fields) != len(self.ax1):
            raise ValueError("Number of fields must match number of axes")
        
        r = Function(self.metamaterial.R)
        for ax, (name, field) in zip(self.ax1, fields.items()):
            if field.size == self.metamaterial.R.dim():
                r.vector()[:] = field
                self.plot_density(r, title=f"{name}", ax=ax)
            else:
                raise ValueError("Field size does not match function space")
            ax.set_xticks([])
            ax.set_yticks([])
            
        self.ax2.clear()
        f_arr = np.asarray(self.evals)
        self.ax2.plot(range(1, len(self.evals)+1), f_arr, marker='o')  
        self.ax2.grid(True)
        self.ax2.set_xlim(left=0, right=len(self.evals) + 2) 
        
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(1e-3)
            
    def plot_density(self, r_in, title=None, ax=None):
        r = Function(r_in.function_space())
        r.vector()[:] = 1. - r_in.vector()[:]
        r.set_allow_extrapolation(True)
        
        if isinstance(ax, plt.Axes):
            plt.sca(ax)
        else:
            fig, ax = plt.subplots()
            
        ax.margins(x=0,y=0)

        # quad meshes aren't supported using the standard plot interface but we can convert them to an image and use imshow
        # the ordering of a quad mesh is row-major and imshow expects row-major so it works out
        cell_type = r_in.function_space().ufl_cell().cellname()
        if cell_type == 'quadrilateral':
            r_vec = r.vector()[:]
            # assume square space
            nely = np.sqrt(r_vec.size).astype(int)
            nelx = nely
            plt.imshow(r_vec.reshape((nely, nelx)), cmap='gray', vmin=0, vmax=1)
            ax.set_title(title)
            return
        
        plot(r, cmap='gray', vmin=0, vmax=1, title=title)
        

class VectorConstraint:
    '''
    This is a generalized class that will handle vector constraints for nlopt.
    It can also be used for a scalar constraint with vector of length 1.
    
    A constraint is formulated as g_k(x) <= b_k(x).
    We rearrange to give nlopt the form g_k(x) - b_k(x) <= 0.
    The simplified case is where b_k(x) = 0.
    This is the assumed default b/c we set dbdt = 0. on initialization.

    One thing we may wish to do is in a minimax problem we can bound using the optimization variable `t`.
    The interface for constraints and bounds takes this into account, but it could be more general in the future.
    e.g. right now the bounds functions are assumed to only take t, but something like b_k(Chom, x, t) and discarding unused inputs is viable.

    We use a global Chom, r and vjp here because it prevents us from having to rerun the forward pass again for the constraint.
    If we wrapped everything into one large class this could just be a class attribute that gets passed around.
    '''
    def __init__(self, ops, eps=1e-6, verbose=False):
        self.ops = ops
        self.constraints = []
        self.eps = eps
        self.verbose = verbose
        self.dbdt = np.zeros(self.n_constraints)
        self.bounds = None
            
    def __call__(self, results, x, grad, dummy_run=False):

        x, t = x[:-1], x[-1]
        
        m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
        Chom = m @ jnp.asarray(self.ops.Chom) @ m
        dChom_dxfem = self.ops.dChom_dxfem
        dxfem_dx_vjp = self.ops.dxfem_dx_vjp
        
        # values
        gs = np.array([g(Chom) for g in self.constraints])
        bs = np.array([b(t) for b in self.bounds])
        dg_dChoms = [jax.jacrev(g)(Chom) for g in self.constraints]
                
        results[:] = gs - bs
        
        if dummy_run:
            return
        
        if self.verbose:
            print(f"{self.__str__()} value(s): {gs}")
            print(f"{self.__str__()} bound(s): {bs}")
        
        if grad.size > 0:
            for n in range(self.n_constraints):
                grad[n,:-1] = dxfem_dx_vjp(dg_dChoms[n].flatten() @ dChom_dxfem)[0]
                grad[n,-1] = -self.dbdt[n]
                    
    @property
    def n_constraints(self):
        return len(self.constraints)
    
    def __str__(self):
        return "MinimaxConstraints"
    
class EigenvectorConstraint:
    
    def __init__(self, v, ops, eps=1e-3, verbose=True):
        self.v = v
        self.ops = ops
        self.eps = eps
        self.verbose = verbose
        
    def __call__(self, x, grad):
        
        Chom, dChom_dxfem, dxfem_dx_vjp = self.ops.Chom, self.ops.dChom_dxfem, self.ops.dxfem_dx_vjp
        
        # def obj(C):
        #     m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
        #     C = m @ C @ m
        #     C /= jnp.linalg.norm(C, ord='fro')
        #     eig_vs = jnp.linalg.eigh(C)[1]
        #     print('My v:', self.v)
        #     print('Eigenvectors:',  jnp.asarray(eig_vs))
        #     return jnp.linalg.norm(eig_vs - self.v)

        def obj(C):
            m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
            C = m @ C @ m
            C /= jnp.linalg.norm(C, ord='fro')
            
            vCv = self.v.T @ C @ self.v
            r = C @ self.v[:,0] - vCv[0,0] * self.v[:,0]
            return jnp.linalg.norm(r)
        
        c, dc_dChom = jax.value_and_grad(obj)(jnp.asarray(Chom))
        
        if grad.size > 0:
            grad[:-1] = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]
            grad[-1] = 0.
            
        if self.verbose:
            print(f"- Eigenvector Constraint: {c:.2e} (Target ≤{self.eps:}) [{'Satisfied' if c <= self.eps else 'Not Satisfied'}]")

            
        return float(c) - self.eps
        
        

class OffDiagonalConstraint(VectorConstraint):

    def __init__(self, v, **kwargs):
        super().__init__(**kwargs)
        self.v = v
        self.constraints = [self.g1]
        self.bounds = [lambda t: self.eps * t]
        self.dbdt = np.ones(self.n_constraints) * self.eps

    def g1(self, C):
        m = jnp.diag(jnp.array([1., 1., np.sqrt(2)]))
        C = m @ C @ m
        C /= jnp.linalg.norm(C, ord=2)
        vCv = self.v.T @ C @ self.v
        return jnp.linalg.norm(vCv - jnp.diag(jnp.diag(vCv)))

    def __str__(self):
        return "OffDiagonalConstraint"

class MaterialSymmetryConstraints(VectorConstraint):
    """
    Enforcing material symmetry can be done in a number of ways.
    For example, Andreassen et al. used one scalar constraint on the whole homogenized matrix, but here we implement each constraint as a separate function.
    
    The terminology and constraints are based on Trageser and Seleson paper.
    https://csmd.ornl.gov/highlight/anisotropic-two-dimensional-plane-strain-and-plane-stress-models-classical-linear
    """
    def __init__(self, symmetry_order='oblique', **kwargs):
        super().__init__(**kwargs)
        self.symmetry_order = symmetry_order
        
        self.symmetry_types_ = ['oblique', 'rectangular', 'square', 'isotropic']
        
        if self.symmetry_order not in self.symmetry_types_:
            raise ValueError(f"Material symmetry must be one of {self.symmetry_types_}")
        
        if self.symmetry_order in self.symmetry_types_[1:]:
            self.constraints.extend([lambda C: (C[0, 2]/C[0, 0])**2, 
                                     lambda C: (C[1, 2]/C[1, 1])**2])
        if self.symmetry_order in self.symmetry_types_[2:]:
            self.constraints.append(lambda C: (1. - C[1, 1]/C[0, 0])**2)
        if self.symmetry_order == 'isotropic':
            self.constraints.append(lambda C: (1. - C[0, 1]/C[0, 0] - C[2, 2]/C[0, 0])**2)
        
        # self.bounds = self.n_constraints * [lambda t: self.eps]
        # self.dbdt = np.zeros(self.n_constraints)
        self.bounds = self.n_constraints * [lambda t: self.eps * t]
        self.dbdt = np.ones(self.n_constraints) * self.eps
        
    def __str__(self):
        return f"MaterialSymmetryConstraints_{self.symmetry_order}"       

class IsotropicConstraint:
    
    def __init__(self, eps, ops, verbose=True):
        self.eps = eps
        self.ops = ops
        self.verbose = verbose
    
    def __call__(self, x, grad):

        Chom = self.ops.Chom
        dChom_dxfem = self.ops.dChom_dxfem
        dxfem_dx_vjp = self.ops.dxfem_dx_vjp
        
        def g(C):
            C = jnp.array(C)
            Ciso = self.compute_Ciso(C)
            diff = Ciso - C
            return jnp.sum(diff**2) / Ciso[0,0]
        
        c, dc_dChom = jax.value_and_grad(g)(Chom)

        if grad.size > 0:
            grad[:] = dxfem_dx_vjp(np.asarray(dc_dChom).flatten() @ dChom_dxfem)[0]
        
        if self.verbose == True:
            print(f"- Isotropic Constraint: {c:.2e} (Target ≤{self.eps:}) [{'Satisfied' if c <= self.eps else 'Not Satisfied'}]")

        return float(c) - self.eps

    def compute_Ciso(self, C):
        Ciso = jnp.zeros_like(C)
        Ciso = Ciso.at[0,1].set(C[0,1])
        Ciso = Ciso.at[1,0].set(C[1,0])
        avg = (C[0,0] + C[1,1]) / 2.
        Ciso = Ciso.at[0,0].set(avg)
        Ciso = Ciso.at[1,1].set(avg)
        Ciso = Ciso.at[2,2].set((Ciso[0,0] - Ciso[0,1]) / 2.)
        return Ciso

class EpigraphBulkModulusConstraint:
    
    def __init__(self, base_E, base_nu, a, ops, verbose=True):
        self.base_E = base_E
        self.base_nu = base_nu
        self.base_K = self.compute_K(self.base_E, self.base_nu)
        self.a = a
        self.aK = self.base_K * self.a
        self.ops = ops
        self.verbose = verbose
        self.n_constraints = 1
        
    def __call__(self, x, grad):

        Chom = self.ops.Chom
        dChom_dxfem = self.ops.dChom_dxfem
        dxfem_dx_vjp = self.ops.dxfem_dx_vjp
        
        # g = lambda C: -0.5 * (C[0][0] + C[1][0])
        def g(C):
            S = jnp.linalg.inv(C)
            return -1. / (S[0][0] + S[0][1]) / 2.
        c, dc_dChom = jax.value_and_grad(g)(jnp.asarray(Chom))
        
        if grad.size > 0:
            grad[:-1] = dxfem_dx_vjp(np.asarray(dc_dChom).flatten() @ dChom_dxfem)[0]
            grad[-1] = 0.

        if self.verbose == True:
            print(f"- Bulk Modulus: {-c:.2e} (Target ≥{self.aK:.2e}) [{'Satisfied' if -c >= self.aK else 'Not Satisfied'}]")
#≤

        return self.aK + float(c)
            
            
    def compute_K(self, E, nu):
        # computes plane stress bulk modulus from E and nu
        K = E / (3 * (1 - 2 * nu))
        G = E / (2 * (1 + nu))
        K_plane = 9.*K*G / (3.*K + 4.*G)
        return K_plane

class BulkModulusConstraint:
    
    def __init__(self, base_E, base_nu, a, ops, verbose=True):
        self.base_E = base_E
        self.base_nu = base_nu
        self.base_K = self.compute_K(self.base_E, self.base_nu)
        self.a = a
        self.aK = self.base_K * self.a
        self.ops = ops
        self.verbose = verbose
        
    def __call__(self, x, grad):

        Chom = self.ops.Chom
        dChom_dxfem = self.ops.dChom_dxfem
        dxfem_dx_vjp = self.ops.dxfem_dx_vjp
        
        # g = lambda C: -0.5 * (C[0][0] + C[1][0])
        def g(C):
            S = jnp.linalg.inv(C)
            return -1. / (S[0][0] + S[0][1]) / 2.
        c, dc_dChom = jax.value_and_grad(g)(jnp.asarray(Chom))
        
        if grad.size > 0:
            grad[:] = dxfem_dx_vjp(np.asarray(dc_dChom).flatten() @ dChom_dxfem)[0]

        if self.verbose == True:
            print(f"- Bulk Modulus: {-c:.2e} (Target ≥{self.aK:.2e}) [{'Satisfied' if -c >= self.aK else 'Not Satisfied'}]")
#≤

        return self.aK + float(c)
            
            
    def compute_K(self, E, nu):
        # computes plane stress bulk modulus from E and nu
        K = E / (3 * (1 - 2 * nu))
        G = E / (2 * (1 + nu))
        K_plane = 9.*K*G / (3.*K + 4.*G)
        return K_plane

class ShearModulusConstraint:
    
    def __init__(self, E_max, nu, ops, a=0.002, verbose=True):
        self.E_max = E_max
        self.nu = nu
        self.G_max = E_max / (2 * (1 + nu))
        self.a = a
        self.aG = self.G_max * self.a

        self.ops = ops
        self.verbose = verbose
        
    def __call__(self, x, grad):

        Chom = self.ops.Chom
        dChom_dxfem = self.ops.dChom_dxfem
        dxfem_dx_vjp = self.ops.dxfem_dx_vjp
        
        # g = lambda C: -C[2][2]
        def g(C):
            S = jnp.linalg.inv(C)
            return -1/S[2][2]
        c, dc_dChom = jax.value_and_grad(g)(jnp.asarray(Chom))
        
        if grad.size > 0:
            grad[:] = dxfem_dx_vjp(np.asarray(dc_dChom).flatten() @ dChom_dxfem)[0]
            
        if self.verbose == True:
            print(f"- Shear Modulus: {-c:.2e} (Target ≥{self.aG:.2e}) [{'Satisfied' if -c >= self.aG else 'Not Satisfied'}]")
        
        return self.aG + float(c)
            
class VolumeConstraint:
    
    def __init__(self, V, ops, verbose=True):
        self.V = V
        self.evals = []
        self.ops = ops
        self.verbose = verbose
        
    def __call__(self, x, grad):
        
        # x_fem = self.ops.x_fem
        filt = self.ops.filt
        beta = self.ops.beta
        eta = self.ops.eta

        # we only constrain the volume of the projected density field per Wang et al. 2011. Right now x_fem sometimes I have a SIMP applied to it, so we do our our filter and projection here. If we remove the SIMP in the Objective function in the future we could use this commented out code b/c x_fem final step would be just the projection
        # x_fem = self.ops.x_fem
        # volume, dvdx = jax.value_and_grad(lambda x: jnp.mean(x))(x_fem)


        def g(x):
            x = jax_density_filter(x, filt.H_jax, filt.Hs_jax)
            x = jax_projection(x, beta, eta)
            return jnp.mean(x)
        
        volume, dvdx = jax.value_and_grad(g)(x)
                
        if grad.size > 0:
            grad[:] = dvdx
            
        if self.verbose == True: 
            print(f"- Volume: {volume:.3f} (Target ≤{self.V}) [{'Satisfied' if volume <= self.V else 'Not Satisfied'}]")
        
        return float(volume) - self.V


class XiaOptimization:
    
    def __init__(self, metamaterial, filt, pen, vol_frac, objective='bulk', max_eval=100):
        self.metamaterial = metamaterial
        self.filt = filt
        self.pen = pen
        self.vol_frac = vol_frac
                
        self.evals = []
        self.max_eval = max_eval
        if objective == 'shear':
            self.obj_idxs = [[2], [2]]
        elif objective == 'bulk':
            self.obj_idxs = [[0,1], [0, 1]]
        else:
            raise ValueError("Invalid objective. Must be 'shear' or 'bulk'")

        
    def optimize(self, x):
        change = 1.
        x_phys = np.copy(x)
        for n in range(self.max_eval):
            if change < 0.01:
                break
            
            self.metamaterial.x.vector()[:] = x_phys**self.pen
            sols, Chom, uChom = self.metamaterial.solve()

            E_max = self.metamaterial.prop.E_max
            E_min = self.metamaterial.prop.E_min
            nu    = self.metamaterial.prop.nu
            cell_vol = next(cells(self.metamaterial.mesh)).volume()
            baseUChom = self.metamaterial.homogenized_C(sols, E_max, nu)[1]

            c = sum(-Chom[i][j] for i in self.obj_idxs[0] for j in self.obj_idxs[1])
            dc = sum(baseUChom[i][j] for i in self.obj_idxs[0] for j in self.obj_idxs[1])
            
            self.evals.append(c)
            dc = -self.pen * (E_max - E_min) * x_phys**(self.pen - 1.) * project(dc, self.metamaterial.R).vector()[:]
            dv = np.ones_like(x) * cell_vol
            
            ft = 'density'
            if ft == 'sensitivity':
                dc = np.divide(self.filt.H @ np.multiply(x, dc), 
                               np.multiply(np.maximum(1e-3, x), self.filt.Hs))
            elif ft == 'density':
                dc = self.filt.H @ np.divide(dc, self.filt.Hs)
                dv = self.filt.H @ np.divide(dv, self.filt.Hs)
            else:
                raise ValueError("Invalid update type")
            
            l1, l2, move = 0., 1e9, 0.2
            while l2 - l1 > 1e-9:
                l_mid = 0.5 * (l1 + l2)
                x_new = np.maximum(0., np.maximum(x - move, np.minimum(1., np.minimum(x + move, x * np.sqrt(-dc / dv / l_mid)))))
                l1, l2 = (l_mid, l2) if np.mean(x_new) - self.vol_frac > 0 else (l1, l_mid)
                if ft == 'sensitivity':
                    x_phys = x_new
                elif ft == 'density':
                    x_phys = self.filt.H @ np.divide(x_new, self.filt.Hs)
                else:
                    raise ValueError("Invalid update type")
                
                if np.mean(x_phys) > self.vol_frac:
                    l1 = l_mid
                else:
                    l2 = l_mid
            
            change = np.linalg.norm(x_new - x, np.inf)
            
            x = x_new
            
            print(f"Step {n+1:d}, Objective = {c:.4f}, Vol = {np.mean(x_phys):.4f} Change = {change:.4f}")
            
        return x_phys

@dataclass
class OptimizationState:
    sols: list = field(default_factory=list)
    Chom: np.array = field(default_factory=lambda: np.zeros((3, 3)))
    dChom_dxfem: np.array = field(default_factory=lambda: np.zeros((3, 3, 1)))
    dxfem_dx_vjp: Callable[[np.ndarray], np.ndarray] = None
    xfem: np.array = field(default_factory=lambda: np.zeros(1))
    beta: float = 1.
    eta:  float = 0.5
    pen:  float = 3. # holdover from other optimization types that use SIMP
    filt: DensityFilter = None
    epoch: int = 0
    epoch_iter_tracker: list = field(default_factory=list)

    def update_state(self, sols, Chom, dChom_dxfem, dxfem_dx_vjp, x_fem):
        self.sols = sols
        self.Chom = Chom
        self.dChom_dxfem = dChom_dxfem
        self.dxfem_dx_vjp = dxfem_dx_vjp
        self.x_fem = x_fem