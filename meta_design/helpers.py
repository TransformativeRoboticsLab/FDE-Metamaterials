import fenics as fe

class Circle(fe.UserExpression):
    
    def __init__(self, V, rad):
        super().__init__()
        self.V = V
        self.rad = rad
        
    def eval(self, values, x):
        xc, yc, r = 0.5, 0.5, self.rad
        values[0] = 0.5*self.V if (x[0] - xc)**2 + (x[1] - yc)**2 < r**2 else self.V
