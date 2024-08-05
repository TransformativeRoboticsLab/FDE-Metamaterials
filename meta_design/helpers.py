import fenics as fe

class Circle(fe.UserExpression):
    
    def __init__(self, V, rad):
        super().__init__()
        self.V = V
        self.rad = rad
        
    def eval(self, values, x):
        xc, yc, r = 0.5, 0.5, self.rad
        values[0] = 0.5*self.V if (x[0] - xc)**2 + (x[1] - yc)**2 < r**2 else self.V

        
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