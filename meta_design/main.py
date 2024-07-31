import fenics as fe
import numpy as np
import nlopt
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from metamaterial import Metamaterial
from filter import DensityFilter

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
        

class PoissonRatio:
    def __init__(self, metamaterial, filt, beta, eta):
        self.metamaterial = metamaterial
        self.filt = filt
        self.beta = beta
        self.eta  = eta
        self.epoch = 0
        self.evals = []
        self.epoch_iter_tracker = []
        
        # self.fig, self.ax = self.metamaterial.plot_density()
        # plt.ion()
        # plt.show()
        
    def __call__(self, x, grad):
        # global global_state
        
        def filter_and_project(x):
            x_tilde = jax_density_filter(x, self.filt.H_jax, self.filt.Hs_jax)
            x_bar   = jax_projection(x_tilde, self.beta, self.eta)
            return x_bar
        
        x_out, vjp_fn = jax.vjp(filter_and_project, x)
        
        # self.metamaterial.x.vector().set_local(x_out)
        self.metamaterial.x.vector()[:] = x_out
        
        sols, Chom = self.metamaterial.solve()
        
        # global_state = (sols, Chom)
        
        E_max = self.metamaterial.prop.E_max
        E_min = self.metamaterial.prop.E_min
        nu    = self.metamaterial.prop.nu
        _, baseUChom = self.metamaterial.homogenized_C(sols, E_max, nu)
        
        val  = -Chom[2][2]
        
        self.evals.append(val)

        if grad.size > 0:
            dvdx = -fe.project(baseUChom[2][2], self.metamaterial.R).vector().get_local() * (E_max - E_min)
            grad[:] = vjp_fn(dvdx)[0]
            
            
        # if len(self.evals) % 10 == 1:
            # self.update_plot()
            
        # self.ax.clear()
        # self.metamaterial.plot_density(self.ax)
        # plt.draw()
        # plt.pause(1e-3)
        
        print(f"Epoch {self.epoch}, Step {len(self.evals)}, C_22 = {-1.*val}")

        return val
    
    
def main():
    nelx = 40
    nely = nelx
    E_max = 1.
    E_min = 1e-9
    nu = 0.3
    vol_frac = 0.35
    
    # metamaterial construction
    metamate = Metamaterial(E_max, E_min, nu)
    metamate.mesh = fe.UnitSquareMesh(nelx, nely, 'crossed')
    W, R = metamate.create_function_spaces()

    # filter
    filt = DensityFilter(metamate.mesh, 0.1, distance_method='periodic')
    
    # initial rho
    x = jnp.array(np.random.binomial(1, vol_frac, R.dim()), dtype=np.float32)

    # optimizer
    f = PoissonRatio(metamaterial=metamate, filt=filt, beta=1, eta=0.5)
    g = VolumeConstraint(V=vol_frac, filt=filt, beta=1, eta=0.5)

    dim = metamate.x.vector().size()
    opt = nlopt.opt(nlopt.LD_MMA, dim)
    opt.set_min_objective(f)
    opt.add_inequality_constraint(g, 0.)
    
    opt.set_lower_bounds(np.zeros(dim))
    opt.set_upper_bounds(np.ones(dim))
    # opt.set_maxeval(100)
    
    opt.optimize(x)
    
    
    
    # sols, Chom = metamate.solve()
    # print(np.array(Chom))
    
    # baseChom, baseUChom = metamate.homogenized_C(sols, E_max, nu)
    # dd = fe.derivative(baseUChom[2][2], metamate.x)
    # print(np.array(baseChom))
    
    
    
    
    # return
    
    # pr = PoissonRatio(metamaterial=metamate, filt=filt, beta=1, eta=0.5)
    
    # pr(x, [])
    
    # S = np.array([[1., -nu, 0],
    #               [-nu, 1., 0.],
    #               [0., 0., 2*(1+nu)]]) / vol_frac
    
    # C = np.linalg.inv(S)
    # print(C)
    
    
    # the flow of the program
    # what should the metamaterial do?
    # what should the optimizer do?
    
    
    # initial rho
    # x = fe.Function(R)
    # x.vector()[:] = np.random.binomial(1, vol_frac, R.dim())
    
    # metamate.x = filt.filter(x)
    
    # sols, Chom = metamate.solve()
    
    
    # def objective(x, grad):
        


if __name__ == "__main__":
    np.random.seed(1)
    main()
