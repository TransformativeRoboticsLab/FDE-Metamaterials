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
        c, dc_dChom = jax.value_and_grad(g)(Chom)
        
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
    pen:  float = 3.
    filt: DensityFilter = None

    def update_state(self, sols, Chom, dChom_dxfem, dxfem_dx_vjp, x_fem):
        self.sols = sols
        self.Chom = Chom
        self.dChom_dxfem = dChom_dxfem
        self.dxfem_dx_vjp = dxfem_dx_vjp
        self.x_fem = x_fem