from dataclasses import dataclass, field

import fenics as fe
import numpy as np
from matplotlib import pyplot as plt

from .boundary import PeriodicDomain
from .mechanics import (lame_parameters, linear_strain, linear_stress,
                        macro_strain)

fe.set_log_level(40)

def setup_metamaterial(E_max, E_min, nu, nelx, nely, mesh_cell_type='triangle', domain_shape='square'):
    metamaterial = Metamaterial(E_max, E_min, nu, nelx, nely, domain_shape=domain_shape)
    if 'tri' in mesh_cell_type:
        P0 = fe.Point(0, 0)
        P1 = fe.Point(1, 1)
        if 'rect' in domain_shape:
            P1 = fe.Point(np.sqrt(3), 1)
            nelx = int(nelx * np.sqrt(3))
            print(f"Rectangular domain requested. Adjusting nelx to {nelx:d} cells to better match aspect ratio.")
        metamaterial.mesh = fe.RectangleMesh(P0, P1, nelx, nely, 'crossed')
        metamaterial.domain_shape = domain_shape
    elif 'quad' in mesh_cell_type:
        metamaterial.mesh = fe.RectangleMesh.create([fe.Point(0, 0), fe.Point(1, 1)],
                                                 [nelx, nely],
                                                 fe.CellType.Type.quadrilateral)
    else:
        raise ValueError(f"Invalid cell_type: {mesh_cell_type}")
    metamaterial.create_function_spaces()
    return metamaterial

class Metamaterial:
    def __init__(self, E_max, E_min, nu, nelx, nely, mesh=None, x=None, domain_shape=None):
        self.prop = Properties(E_max, E_min, nu)
        self.nelx = nelx
        self.nely = nely
        self.x = x
        self.mesh = mesh
        self.domain_shape = domain_shape
        self.mirror_map = None

    def plot_mesh(self, labels=False, ):
        if self.mesh.ufl_cell().cellname() == 'quadrilateral':
            print("Quadrilateral mesh plotting not supported")
            plt.figure()
            return

        fe.plot(self.mesh)
        if labels:
            for c in fe.cells(self.mesh):
                plt.text(c.midpoint().x(), c.midpoint().y(), str(c.index()))

            # plt.scatter([m.x() for m in mids], [m.y()
                        # for m in mids], marker='x', color='red')

        plt.show(block=True)

    def plot_density(self):
        # if isinstance(self.mesh.)
        r = fe.Function(self.R)
        r.vector()[:] = 1. - self.x.vector()[:]
        r.set_allow_extrapolation(True)

        title = f"Density - Average {np.mean(self.x.vector()[:]):.3f}"
        plt.figure()
        fe.plot(r, cmap='gray', vmin=0, vmax=1, title=title)
        plt.show(block=True)

    def create_function_spaces(self, elem_degree=1):
        assert elem_degree >= 1, "Element degree must be at least 1"
        if not isinstance(self.mesh, fe.Mesh):
            raise ValueError("self.mesh is not a valid mesh")
        PBC = PeriodicDomain(self.mesh)
        Ve = fe.VectorElement('CG', self.mesh.ufl_cell(), elem_degree)
        Re = fe.VectorElement('R', self.mesh.ufl_cell(), 0)
        W = fe.FunctionSpace(self.mesh, fe.MixedElement(
            [Ve, Re]), constrained_domain=PBC)

        # function spaces for rho (R), a continuous version of rho (R_cg), and the gradient of rho (R_grad).
        # R is discontinuous, so we need a continuous space to project to so we can calculate the gradient
        R = fe.FunctionSpace(self.mesh, 'DG', 0, constrained_domain=PBC)
        # R = fe.FunctionSpace(self.mesh, 'CG', 1, constrained_domain=PBC)
        R_cg = fe.FunctionSpace(self.mesh, 'CG', 1, constrained_domain=PBC)
        R_grad = fe.VectorFunctionSpace(self.mesh, 'CG', 1, constrained_domain=PBC)
        R_tri = fe.FunctionSpace(fe.UnitSquareMesh(self.nelx, self.nely, 'crossed'), 'DG', 0)

        self.x = fe.Function(R)
        self.PBC = PBC
        self.W = W
        self.R, self.R_cg, self.R_grad = R, R_cg, R_grad
        self.R_tri = R_tri

    def _project_uChom_to_matrix(self, uChom):
        projected_values = np.empty((9, self.R.dim()))  # Preallocate the array

        for idx, (i, j) in enumerate(((i, j) for i in range(3) for j in range(3))):
            projected_function = fe.project(uChom[i][j], self.R)
            projected_values[idx, :] = projected_function.vector().get_local()

        return projected_values
 

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
        # Chom = [[assemble(uChom[i][j]*dx) for j in range(3)] for i in range(3)]

        # Must scale by cell volume because we aren't having ics account for that in the background
        # note: this makes the assumption that the mesh is uniform
        # note note: we can also sum up these rows to get our Chom, which is the same as doing the "assembly"
        # summing the values is faster than the assembly, and since we have to make the uChom matrix anyway we might as well do it this way.
        # if we don't need the uChom matrix, the doing assemble might be faster again

        uChom_matrix = self._project_uChom_to_matrix(uChom) * self.cell_vol / self.domain_vol
        # remember the matrix is symmetric so we don't care about row/column order
        Chom = np.reshape(np.sum(uChom_matrix, axis=1), (3,3))

        return Chom, uChom_matrix

    def solve(self):
        v_, lamb_ = fe.TestFunctions(self.W)
        dv, dlamb = fe.TrialFunctions(self.W)

        E = self.prop.E_min + (self.prop.E_max - self.prop.E_min) * self.x
        nu = self.prop.nu

        m_strain = fe.Constant(((0., 0.),
                             (0., 0.)))
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

    @property
    def cell_vol(self):
        return next(fe.cells(self.mesh)).volume()

    @property
    def resolution(self):
        x_min, y_min = self.mesh.coordinates().min(axis=0)
        x_max, y_max = self.mesh.coordinates().max(axis=0)
        return (x_max - x_min) / self.nelx, (y_max - y_min) / self.nely

    @property
    def domain_vol(self):
        return fe.assemble(fe.Constant(1)*fe.dx(domain=self.mesh))
    
    @property
    def width(self):
        return self.resolution[0] * self.nelx
    
    @property
    def height(self):
        return self.resolution[1] * self.nely

    @property
    def cell_midpoints(self):
        return np.array([c.midpoint().array()[:2] for c in fe.cells(self.mesh)])


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
