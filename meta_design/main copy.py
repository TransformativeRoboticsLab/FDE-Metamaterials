from fenics import *
from fenics_adjoint import *
import numpy as np
import nlopt
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from metamaterial import Metamaterial
from filters import DensityFilter

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

class Circle(UserExpression):
    def eval(self, values, x):
        xc, yc, rad = 0.5, 0.5, 0.25
        values[0] = 1e-3 if (x[0] - xc)**2 + (x[1] - yc)**2 < rad**2 else .35

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
    def __init__(self, metamaterial, filt, beta, eta, plot=True, filter_and_project=True):
        self.metamaterial = metamaterial
        self.filt = filt
        self.beta = beta
        self.eta  = eta
        self.plot = plot

        self.epoch = 0
        self.evals = []
        self.epoch_iter_tracker = []
        self.filter_and_project = filter_and_project
        
    def __call__(self, x, grad):
        # global global_state
        
        def filter_and_project(x):
            if not self.filter_and_project:
                return x
            x_tilde = jax_density_filter(x, self.filt.H_jax, self.filt.Hs_jax)
            x_bar   = jax_projection(x_tilde, self.beta, self.eta)
            return x_bar
        
        x_out, vjp_fn = jax.vjp(filter_and_project, x)
        
        self.metamaterial.x.vector()[:] = x_out
        
        sols, Chom = self.metamaterial.solve()
        
        # global_state = (sols, Chom)
        
        E_max = self.metamaterial.prop.E_max
        E_min = self.metamaterial.prop.E_min
        nu    = self.metamaterial.prop.nu
        _, baseUChom = self.metamaterial.homogenized_C(sols, Constant(E_max), Constant(nu))
        
        val  = -Chom[2][2]
        cell_vol = next(cells(self.metamaterial.mesh)).volume()
        dvdx = project(-baseUChom[2][2] * cell_vol, self.metamaterial.R)# * self.metamaterial.x
        dd = compute_gradient(val, Control(self.metamaterial.x))
        # dvdx = -baseUChom[2][2]

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(18,6))
        plt.sca(ax1)
        h = plot(dvdx)
        plt.colorbar(h)
        plt.sca(ax2)
        h = plot(dd)
        plt.colorbar(h)
        plt.sca(ax3)
        h = plot(dd - dvdx)
        plt.colorbar(h)
        a1 = dd.vector().norm('l2')
        a2 = dvdx.vector().norm('l2')
        print(f"Err: {(a1-a2)/a1:.2f}")
        # print(dd.vector()[:] / dvdx.vector()[:])
        

        
        plt.show(block=True)
        self.evals.append(val)

        if grad.size > 0:
            # dvdx = compute_gradient(val, Control(self.metamaterial.x)).vector().get_local()
            # dvdx = project(-baseUChom[2][2], self.metamaterial.R).vector().get_local()
            dvdx = dd
            grad[:] = np.array(vjp_fn(dvdx)[0])
            
        if (len(self.evals) % 10 == 1) and self.plot:
            self.metamaterial.plot_density(title=f"Density - Iter: {len(self.evals):d}")
            
        
        print(f"Epoch {self.epoch}, Step {len(self.evals)}, C_22 = {-1.*val}")

        return val
    
    
def main():
    nelx = 40
    nely = nelx
    E_max = 1.
    E_min = 1e-9
    nu = 0.3
    vol_frac = 0.35

    beta = 16
    eta = 0.5
    
    # metamaterial construction
    metamate = Metamaterial(E_max, E_min, nu)
    metamate.mesh = UnitSquareMesh(nelx, nely, 'crossed')
    # print(assemble(Constant(1)*dx(metamate.mesh)))
    W, R = metamate.create_function_spaces()

    # filter
    filt = DensityFilter(metamate.mesh, 0.1, distance_method='periodic')
    
    # initial rho
    x = jnp.array(np.random.binomial(1, vol_frac, R.dim()), dtype=np.float32)
    # x = np.ones(R.dim())
    # r = Function(metamate.R)
    # r.assign(interpolate(Circle(), metamate.R))
    # x = r.vector()[:]

    # optimizer
    f = PoissonRatio(metamaterial=metamate, filt=filt, beta=beta, eta=eta, filter_and_project=True)
    f(x, np.array([]))
    # g = VolumeConstraint(V=vol_frac, filt=filt, beta=beta, eta=eta)

    # dim = metamate.x.vector().size()
    # opt = nlopt.opt(nlopt.LD_MMA, dim)
    # opt.set_min_objective(f)
    # opt.add_inequality_constraint(g, 0.)
    
    # opt.set_lower_bounds(np.zeros(dim))
    # opt.set_upper_bounds(np.ones(dim))
    # # opt.set_maxeval(100)
    
    # opt.optimize(x)



if __name__ == "__main__":
    np.random.seed(1)
    main()
