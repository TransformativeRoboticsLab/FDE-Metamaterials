from boundary import PeriodicDomain
import numpy as np
from mechanics import linear_strain, linear_stress, macro_strain, lame_parameters
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
from fenics import *
set_log_level(40)


class Metamaterial:
    def __init__(self, E_max, E_min, nu, nelx, nely, mesh=None, x=None):
        self.prop = Properties(E_max, E_min, nu)
        self.nelx = nelx
        self.nely = nely
        self.x = x
        self.mesh = mesh

    def plot_mesh(self, labels=False, ):
        if self.mesh.ufl_cell().cellname() == 'quadrilateral':
            print("Quadrilateral mesh plotting not supported")
            plt.figure()
            return

        plot(self.mesh)
        if labels:
            mids = [cell.midpoint() for cell in cells(self.mesh)]
            ids = [cell.index() for cell in cells(self.mesh)]
            for mid, id in zip(mids, ids):
                plt.text(mid.x(), mid.y(), str(id))

            plt.scatter([m.x() for m in mids], [m.y()
                        for m in mids], marker='x', color='red')

        plt.show(block=True)

    def plot_density(self):
        # if isinstance(self.mesh.)
        r = Function(self.R)
        r.vector()[:] = 1. - self.x.vector()[:]
        r.set_allow_extrapolation(True)

        title = f"Density - Average {np.mean(self.x.vector()[:]):.3f}"
        plt.figure()
        plot(r, cmap='gray', vmin=0, vmax=1, title=title)
        plt.show(block=True)

    def create_function_spaces(self, elem_degree=1):
        assert elem_degree >= 1, "Element degree must be at least 1"
        # if not isinstance(self.mesh, Mesh):
        # raise ValueError("self.mesh is not a valid mesh")
        PBC = PeriodicDomain(self.mesh)
        Ve = VectorElement('CG', self.mesh.ufl_cell(), elem_degree)
        Re = VectorElement('R', self.mesh.ufl_cell(), 0)
        W = FunctionSpace(self.mesh, MixedElement(
            [Ve, Re]), constrained_domain=PBC)

        # function spaces for rho (R), a continuous version of rho (R_cg), and the gradient of rho (R_grad).
        # R is discontinuous, so we need a continuous space to project to so we can calculate the gradient
        R = FunctionSpace(self.mesh, 'DG', 0, constrained_domain=PBC)
        R_cg = FunctionSpace(self.mesh, 'CG', 2, constrained_domain=PBC)
        R_grad = VectorFunctionSpace(self.mesh, 'CG', 2, constrained_domain=PBC)

        self.x = Function(R)
        self.PBC = PBC
        self.W = W
        self.R, self.R_cg, self.R_grad = R, R_cg, R_grad

    def _project_uChom_to_matrix(self, uChom):
        projected_values = []

        for i in range(3):
            for j in range(3):
                projected_function = project(uChom[i][j], self.R)
                projected_values.append(
                    projected_function.vector().get_local())

        matrix = np.array(projected_values)

        return matrix

    def homogenized_C(self, u_list, E, nu):
        s_list = [linear_stress(linear_strain(u) + macro_strain(i), E, nu)
                  for i, u in enumerate(u_list)]

        uChom = [
            [
                inner(s_t, linear_strain(u) + macro_strain(j))
                for j, u, in enumerate(u_list)
            ]
            for s_t in s_list
        ]
        Chom = [[assemble(uChom[i][j]*dx) for j in range(3)] for i in range(3)]

        # Must scale by cell volume because we aren't having ics account for that in the background
        # note: this makes the assumption that the mesh is uniform
        # note note: we can also sum up these rows to get our Chom, which is the same as doing the "assembly"
        # summing the values is faster than the assembly, and since we have to make the uChom matrix anyway we might as well do it this way.
        # if we don't need the uChom matrix, the doing assemble might be faster again

        uChom_matrix = self._project_uChom_to_matrix(uChom) * self.cell_vol
        # remember the matrix is symmetric so we don't care about row/column order
        # Chom = np.reshape(np.sum(uChom_matrix, axis=1), (3,3))

        return Chom, uChom_matrix

    def solve(self):
        v_, lamb_ = TestFunctions(self.W)
        dv, dlamb = TrialFunctions(self.W)

        E = self.prop.E_min + (self.prop.E_max - self.prop.E_min) * self.x
        nu = self.prop.nu

        m_strain = Constant(((0., 0.),
                             (0., 0.)))
        F = inner(linear_stress(linear_strain(dv) + m_strain, E, nu),
                  linear_strain(v_))*dx
        a, L = lhs(F), rhs(F)
        a += dot(lamb_, dv)*dx + dot(dlamb, v_)*dx

        sols = []
        for (j, case) in enumerate(["Exx", "Eyy", "Exy"]):
            w = Function(self.W)
            m_strain.assign(macro_strain(j))
            solve(a == L, w, [])
            v = split(w.copy(deepcopy=True))[0]
            sols.append(v)

        Chom, uChom = self.homogenized_C(sols, E, nu)

        return sols, Chom, uChom

    @property
    def cell_vol(self):
        return next(cells(self.mesh)).volume()

    @property
    def resolution(self):
        x_min, y_min = self.mesh.coordinates().min(axis=0)
        x_max, y_max = self.mesh.coordinates().max(axis=0)
        return (x_max - x_min) / self.nelx, (y_max - y_min) / self.nely


@dataclass
class Properties:
    E_max: float
    E_min: float
    nu: float
    K: float = field(init=False)
    lambda_: float = field(init=False)
    mu_: float = field(init=False)

    def __post_init__(self):
        self.lambda_, self.mu_ = lame_parameters(
            self.E_max, self.nu, model='plane_stress')
        self.K = self.lambda_ + 2.0*self.mu_
