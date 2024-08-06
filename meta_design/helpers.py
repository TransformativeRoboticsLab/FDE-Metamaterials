import fenics as fe
import numpy as np

class Ellipse(fe.UserExpression):
    
    def __init__(self, V, a, b):
        super().__init__()
        self.V = V
        self.a = a
        self.b = b
        
    def eval(self, values, x):
        xc, yc, a, b = 0.5, 0.5, self.a, self.b
        values[0] = 0.5*self.V if ((x[0] - xc)/a)**2 + ((x[1] - yc)/b)**2 < 1 else self.V
        # values[0] = 0.5*self.V if (x[0] - xc)**2 + (x[1] - yc)**2 < r**2 else self.V

        
def print_summary(optim_type, nelx, nely, E_max, E_min, nu, vol_frac, betas, eta, pen, epoch_duration, a):
    summary = f"""
    Summary of Input Values:
    ------------------------
    optim_type: {optim_type}
    nelx: {nelx}
    nely: {nely}
    E_max: {E_max}
    E_min: {E_min}
    nu: {nu}
    vol_frac: {vol_frac}
    betas: {betas}
    eta: {eta}
    pen: {pen}
    epoch_duration: {epoch_duration}
    a: {a}
    """
    print(summary)