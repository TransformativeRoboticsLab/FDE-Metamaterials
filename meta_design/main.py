import fenics as fe
import numpy as np
import nlopt
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

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

class PoissonRatio:
    def __init__(self, metamaterial, filt, beta=1, eta=0.5):
        self.metamaterial = metamaterial
        self.filt = filt
        self.beta = beta
        self.eta  = eta
        self.epoch = 0
        self.evals = []
        self.epoch_iter_tracker = []
        
    def __call__(self, x, grad):
        global global_state
        
        def filter_and_project(x):
            x_tilde = jax_density_filter(x, self.filt.H_jax, self.filt.Hs_jax)
            x_bar   = jax_projection(x, self.beta, self.eta)
            return x_bar
        
        x_out, vjp = jax.vjp(filter_and_project, x)
        
        self.metamaterial.x.vector().set_local(x_out)
        
        sols, Chom = self.metamaterial.solve()
        
        E_max = self.metamaterial.prop.E_max
        E_min = self.metamaterial.prop.E_min
        nu    = self.metamaterial.prop.nu
        baseChom, baseUChom = self.metamaterial.homogenized_C(sols, E_max, nu)
        
        val  = -Chom[2][2]
        dvdx = -fe.derivative(baseUChom[2][2], self.metamaterial.x) * (E_max - E_min)
        
        if grad.size > 0:
            grad[:] = vjp(dvdx)
            
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
    x = jnp.array(np.random.binomial(1, vol_frac, R.dim()))

    # optimizer
    opt = nlopt.opt(nlopt.LD_MMA, R.dim())
    f = PoissonRatio(metamaterial=metamate, filt=filt, beta=1, eta=0.5)
    opt.set_min_objective(f)
    opt.set_lower_bounds(0.)
    opt.set_upper_bounds(1.)
    
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
    main()
