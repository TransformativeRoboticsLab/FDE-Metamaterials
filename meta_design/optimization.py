import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import fenics as fe
import ufl

@jax.jit
def jax_density_filter(x, H, Hs):
    return jnp.divide(H @ x, Hs)

@jax.jit
def jax_projection(x, beta=1., eta=0.5):
    tanh_beta_eta = jnp.tanh(beta * eta)
    tanh_beta_x_minus_eta = jnp.tanh(beta * (x - eta))
    tanh_beta_one_minus_eta = jnp.tanh(beta * (1 - eta))

    numerator = tanh_beta_eta + tanh_beta_x_minus_eta
    denominator = tanh_beta_eta + tanh_beta_one_minus_eta

    return jnp.array(numerator / denominator)

@jax.jit
def jax_simp(x, penalty):
    return jnp.power(x, penalty)

class Objective:
    def __init__(self, metamaterial, filt, beta, eta, pen, plot=True, filter_and_project=True):
        self.metamaterial = metamaterial
        self.filt = filt
        self.beta = beta
        self.eta  = eta
        self.pen  = pen
        self.plot = plot

        self.epoch = 0
        self.evals = []
        self.epoch_iter_tracker = []
        self.filter_and_project = filter_and_project
        
        if self.plot:
            plt.ion()
            self.fig, self.axes = plt.subplots(1, 3, figsize=(12,4))
        
    def __call__(self, x, grad):
        global global_state
        
        def filter_and_project(x):
            if not self.filter_and_project:
                return x
            x = jax_density_filter(x, self.filt.H_jax, self.filt.Hs_jax)
            x = jax_projection(x, self.beta, self.eta)
            x = jax_simp(x, self.pen)
            return x
        
        x_fem, dxfem_dx_vjp = jax.vjp(filter_and_project, x)
                
        self.metamaterial.x.vector()[:] = x_fem
        
        sols, Chom, uChom = self.metamaterial.solve()
        
        E_max = self.metamaterial.prop.E_max
        nu    = self.metamaterial.prop.nu
        # dChom_dxout is a 3x3 list of lists that are fenics expressions. In a sense this a 3x3xN matrix, where the derivatives are with respect to the Nth component of the design vector.
        # This is achieved by applying the displacement solutions to a material of constant stiffness.
        # Xia has a good description in their paper, eq (22), but this is only for the derivative of the FEM solve. We handle all the other parts of the chain rule with JAX.
        dChom_dxfem = self.metamaterial.homogenized_C(sols, E_max, nu)[1]

        global_state = (sols, Chom, dChom_dxfem, dxfem_dx_vjp)
        
        # obj = lambda C: -(C[0][1] / C[0][0] + C[1][0] / C[1][1])
        # obj = lambda C: -C[2][2]
        # obj = lambda C: -(C[0][0] + C[1][1] + C[0][1] + C[1][0])
        obj = lambda C: -0.5 * (C[0][0] + C[0][1])
            
        c, dc_dChom = jax.value_and_grad(obj)(Chom)
        
        # c = -Chom[0][1] / Chom[1][1] - Chom[1][0] / Chom[1][1]
        # dc = fe.project(ufl.diff(-uChom[0][1] / uChom[1][1] - uChom[1][0] / uChom[1][1], self.metamaterial.x), self.metamaterial.R)
        # dc = fe.project(-dChom_dxout[0][1] / dChom_dxout[1][1] - dChom_dxout[1][0] / dChom_dxout[1][1], self.metamaterial.R)
        
        self.evals.append(c)

        # if grad.size > 0:
        #     g = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]
        #     grad[:] = g
        
        if grad.size > 0:
            g = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]
            grad[:] = g
            
        if (len(self.evals) % 10 == 1) and self.plot:
            fields = {'x': x,
                      'x_tilde': jax_density_filter(x, self.filt.H_jax, self.filt.Hs_jax),
                      'x_fem': x_fem}
            self.update_plot(fields)
            
        
        print(f"Epoch {self.epoch}, Step {len(self.evals)}, C_22 = {c}")

        return float(c)
    
    def update_plot(self, fields):
        if len(fields) != len(self.axes):
            raise ValueError("Number of fields must match number of axes")
        
        r = fe.Function(self.metamaterial.R)
        for ax, (name, field) in zip(self.axes, fields.items()):
            if field.size == self.metamaterial.R.dim():
                r.vector()[:] = field
                self.plot_density(r, title=f"{name}", ax=ax)
            else:
                pass
            ax.set_xticks([])
            ax.set_yticks([])
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(1e-3)
            
    def plot_density(self, r_in, title=None, ax=None):
        r = fe.Function(r_in.function_space())
        r.vector()[:] = 1. - r_in.vector()[:]
        r.set_allow_extrapolation(True)
        
        if isinstance(ax, plt.Axes):
            plt.sca(ax)
        else:
            fig, ax = plt.subplots()
            
        ax.margins(x=0,y=0)
        
        fe.plot(r, cmap='gray', vmin=0, vmax=1, title=title)

class IsotropicConstraint:
    
    def __init__(self, eps=1e-5):
        self.eps = eps
    
    def __call__(self, x, grad):
        global global_state

        sols, Chom, dChom_dxfem, dxfem_dx_vjp = global_state
        
        def fwd(Chom):
            Chom = jnp.array(Chom)
            Ciso = self.compute_Ciso(Chom)
            diff = Ciso - Chom
            return jnp.sum(diff**2) / Ciso[0,0]
        
        c, dc_dChom = jax.value_and_grad(fwd)(Chom)
        
        if grad.size > 0:
            g = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]
            grad[:] = g
            
        print(f"Isotropic Constraint = {c:.2e} <= {self.eps:.2e}")
        return float(c) - self.eps

    def compute_Ciso(self, Chom: jnp.array) -> jnp.array:
        Ciso = jnp.zeros_like(Chom)
        Ciso = Ciso.at[0,1].set(Chom[0,1])
        Ciso = Ciso.at[1,0].set(Chom[1,0])
        avg = (Chom [0,0] + Chom[1,1]) / 2.
        Ciso = Ciso.at[0,0].set(avg)
        Ciso = Ciso.at[1,1].set(avg)
        Ciso = Ciso.at[2,2].set((Ciso[0,0] - Ciso[0,1]) / 2.)
        return Ciso

class BulkModulusConstraint:
    
    def __init__(self, base_E, base_nu, a):
        self.base_E = base_E
        self.base_nu = base_nu
        self.base_K = self.compute_K(self.base_E, self.base_nu)
        self.a = a
        self.aK = self.base_K * self.a
        
    def __call__(self, x, grad):
        global global_state
        sols, Chom, dChom_dxfem, dxfem_dx_vjp = global_state
        
        g = lambda C: -0.25 * (C[0][0] + C[1][1] + C[0][1] + C[1][0])
        c, dc_dChom = jax.value_and_grad(g)(Chom)
        
        if grad.size > 0:
            g = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]
            grad[:] = g

        print(f"Bulk Modulus Constraint: {-c:.2e} >= {self.aK:.2e}")
        return self.aK + float(c)
            
            
    def compute_K(self, E, nu):
        # computes plane stress bulk modulus from E and nu
        K = E / (3 * (1 - 2 * nu))
        G = E / (2 * (1 + nu))
        K_plane = 9.*K*G / (3.*K + 4.*G)
        return K_plane
            
class VolumeConstraint:
    
    def __init__(self, V, filt, beta, eta):
        self.V = V
        self.filt = filt
        self.beta = beta
        self.eta = eta
        self.evals = []
        
    def __call__(self, x, grad):
        # global global_state
        
        def fwd(x):
            x_tilde = jax_density_filter(x, self.filt.H_jax, self.filt.Hs_jax)
            x_bar   = jax_projection(x_tilde, self.beta, self.eta)
            return jnp.mean(x_bar)
        
        volume, dvdx = jax.value_and_grad(fwd)(x)
                
        if grad.size > 0:
            grad[:] = dvdx
            
        print(f"Volume = {volume:.3f} <= {self.V:.3f}")
        
        return float(volume - self.V)

        
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
            cell_vol = next(fe.cells(self.metamaterial.mesh)).volume()
            baseUChom = self.metamaterial.homogenized_C(sols, E_max, nu)[1]

            c = sum(-Chom[i][j] for i in self.obj_idxs[0] for j in self.obj_idxs[1])
            dc = sum(baseUChom[i][j] for i in self.obj_idxs[0] for j in self.obj_idxs[1])
            
            self.evals.append(c)
            dc = -self.pen * (E_max - E_min) * x_phys**(self.pen - 1.) * fe.project(dc, self.metamaterial.R).vector()[:]
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
                
