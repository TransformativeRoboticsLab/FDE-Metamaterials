import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import fenics as fe

def jax_density_filter(x, H, Hs):
    return jnp.divide(H @ x, Hs)

def jax_projection(x, beta=1., eta=0.5):
    tanh_beta_eta = jnp.tanh(beta * eta)
    tanh_beta_x_minus_eta = jnp.tanh(beta * (x - eta))
    tanh_beta_one_minus_eta = jnp.tanh(beta * (1 - eta))

    numerator = tanh_beta_eta + tanh_beta_x_minus_eta
    denominator = tanh_beta_eta + tanh_beta_one_minus_eta

    return jnp.array(numerator / denominator)

def jax_simp(x, penalty):
    return jnp.power(x, penalty)

class Objective:
    def __init__(self, metamaterial, filt, beta, eta, pen, plot=True, filter_and_project=True):
        self.metamaterial = metamaterial
        self.filt = filt
        self.beta = beta
        self.eta  = eta
        self.plot = plot
        self.pen = pen

        self.epoch = 0
        self.evals = []
        self.epoch_iter_tracker = []
        self.filter_and_project = filter_and_project
        
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(12,4))
        
    def __call__(self, x, grad):
        
        def filter_and_project(x):
            if not self.filter_and_project:
                return x
            x = jax_density_filter(x, self.filt.H_jax, self.filt.Hs_jax)
            # x = jax_projection(x, self.beta, self.eta)
            x = jax_simp(x, self.pen)
            return x
        
        x_out, vjp_fn = jax.vjp(filter_and_project, x)
                
        self.metamaterial.x.vector()[:] = x_out
        
        sols, Chom = self.metamaterial.solve()
        
        # global_state = (sols, Chom)
        
        E_max = self.metamaterial.prop.E_max
        E_min = self.metamaterial.prop.E_min
        nu    = self.metamaterial.prop.nu
        _, baseUChom = self.metamaterial.homogenized_C(sols, E_max, nu)
        
        # val  = -Chom[2][2]
        val = -(Chom[0][0] + Chom[0][1] + Chom[1][0] + Chom[1][1])
        
        self.evals.append(val)

        if grad.size > 0:
            cell_vol = next(fe.cells(self.metamaterial.mesh)).volume()
            dvdx = -(baseUChom[0][0] + baseUChom[1][0] + baseUChom[0][1] + baseUChom[1][1])
            dvdx = fe.project(dvdx * cell_vol * (E_max - E_min) * self.pen * self.metamaterial.x ** (self.pen - 1.), self.metamaterial.R).vector().get_local()
            grad[:] = np.array(vjp_fn(dvdx)[0])
            
        if (len(self.evals) % 10 == 1) and self.plot:
            fields = {'x': x,
                      'x_tilde': jax_density_filter(x, self.filt.H_jax, self.filt.Hs_jax),
                      'x_out': x_out}
            self.update_plot(fields)
            
        
        print(f"Epoch {self.epoch}, Step {len(self.evals)}, C_22 = {-1.*val}")

        return val
    
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
            grad[:] = np.array(dvdx)
            
        print(f"Volume = {volume:.3f} <= {self.V:.3f}")
        
        return float(volume - self.V)