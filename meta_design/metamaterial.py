import fenics as fe
fe.set_log_level(40)
from matplotlib import pyplot as plt
from dataclasses import dataclass, field

from mechanics import *
from boundary import PeriodicDomain

class Metamaterial:
    
    def __init__(self, E_max, E_min, nu):
        self.nelx = None
        self.nely = None
        self.x    = None
        self.prop = Properties(E_max, E_min, nu)
        self.mesh = None
        
    def plot_mesh(self):
        fe.plot(self.mesh)
        plt.show()
        

    def plot_density(self, title=None):
        r = fe.Function(self.R)
        r.vector()[:] = 1. - self.x.vector()[:]
        r.set_allow_extrapolation(True)
        
        fe.plot(r, cmap='gray', vmin=0, vmax=1, title=title)
        plt.show()
        
    def create_function_spaces(self, elem_degree=1):
        if not isinstance(self.mesh, fe.Mesh):
            raise ValueError("self.mesh is not a valid mesh")
        Ve = fe.VectorElement('CG', self.mesh.ufl_cell(), elem_degree)
        Re = fe.VectorElement('R', self.mesh.ufl_cell(), 0)
        W  = fe.FunctionSpace(self.mesh, fe.MixedElement([Ve, Re]), constrained_domain=PeriodicDomain(self.mesh))

        R = fe.FunctionSpace(self.mesh, 'DG', 0)
        
        self.x = fe.Function(R)
        self.W, self.R = W, R

        return W, R

    def project_uChom_to_matrix(self, uChom):
        # Initialize an empty list to store the projected values
        projected_values = []

        # Iterate over the 3x3 list of lists
        for i in range(3):
            for j in range(3):
                # Project each element into the function space R
                projected_function = fe.project(uChom[i][j], self.R)
                # Append the projected values to the list
                projected_values.append(projected_function.vector().get_local())

        # Convert the list of projected values into a 9xN matrix
        matrix = np.array(projected_values)

        return matrix

    def homogenized_C(self, u_list, E, nu):
        s_list = [linear_stress(linear_strain(u) + macro_strain(i), E, nu) 
                for i, u in enumerate(u_list)]
        
        uChom = [
            [
                fe.inner(s_t, linear_strain(u) + macro_strain(j))
                for j, u, in enumerate(u_list)
            ]
            for s_t in s_list
        ]
        
        Chom = [[fe.assemble(uChom[i][j]*fe.dx) for j in range(3)] for i in range(3)]

        # Project uChom to a matrix
        # Must scale by cell volume because we aren't having fenics account for that in the background
        # note: this makes the assumption that the mesh is uniform
        # note note: we could also sum up these rows to get the values for Chom, but idk if its really all that much faster than just assembling the matrix
        cell_vol = next(fe.cells(self.mesh)).volume()
        uChom_matrix = self.project_uChom_to_matrix(uChom) * cell_vol
        
        # # Sum up the values of each row in the matrix
        # summed_values = np.sum(uChom_matrix, axis=1)
        
        # # Compare summed values with Chom
        # for i in range(3):
        #     for j in range(3):
        #         print(f"Chom[{i}][{j}] = {Chom[i][j]}, Summed value = {summed_values[i*3 + j]}, diff={Chom[i][j] - summed_values[i*3 + j]}")
        
        return Chom, uChom_matrix        

    def solve(self):
        v_, lamb_ = fe.TestFunctions(self.W)
        dv, dlamb = fe.TrialFunctions(self.W)
        
        E = self.prop.E_min + (self.prop.E_max - self.prop.E_min) * self.x
        nu = self.prop.nu
        
        m_strain = macro_strain(0)
        F = fe.inner(linear_stress(linear_strain(dv) + m_strain, E, nu), 
                     linear_strain(v_))*fe.dx
        a, L = fe.lhs(F), fe.rhs(F)
        a += fe.dot(lamb_, dv)*fe.dx + fe.dot(dlamb, v_)*fe.dx
        
        sols = []
        for (j, case) in enumerate(["Exx", "Eyy", "Exy"]):
            w = fe.Function(self.W)
            m_strain.assign(macro_strain(j))
            fe.solve(a == L, w, [])
            v = fe.split(w.copy(deepcopy=True))[0]
            sols.append(v)
            
        Chom, uChom = self.homogenized_C(sols, E, nu)
        
        return sols, Chom, uChom

        
@dataclass
class Properties:
    E_max: float
    E_min: float
    nu: float
    K: float = field(init=False)
    lambda_: float = field(init=False)
    mu_: float = field(init=False)

    def __post_init__(self):
        self.lambda_, self.mu_ = lame_parameters(self.E_max, self.nu, model='plane_stress')
        self.K = self.lambda_ + 2.0*self.mu_