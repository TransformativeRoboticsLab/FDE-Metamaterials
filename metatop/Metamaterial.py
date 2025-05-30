import time
from dataclasses import dataclass, field
from functools import cached_property

import fenics as fe
import numpy as np
import numpy.testing as npt
from loguru import logger
from matplotlib import pyplot as plt

from .boundaries import PeriodicDomain
from .fem_profiler import (fem_profiler, profile_assembly,
                           profile_fem_solution, profile_solve)
from .mechanics import (lame_parameters, linear_strain, linear_stress,
                        macro_strain)
from .profiling import ProfileConfig, profile_block, profile_function

fe.set_log_level(40)


def setup_metamaterial(E_max, E_min, nu, nelx, nely, mesh_cell_type='triangle', domain_shape='square', elem_degree=1):
    metamaterial = Metamaterial(
        E_max, E_min, nu, nelx, nely, domain_shape=domain_shape)
    if 'tri' in mesh_cell_type:
        logger.debug("Creating mesh with triangular cells")
        P0 = fe.Point(0, 0)
        P1 = fe.Point(1, 1)
        if 'rect' in domain_shape:
            logger.warning("'rect' domain shape is currently unverified.")
            P1 = fe.Point(np.sqrt(3), 1)
            nelx = int(nelx * np.sqrt(3))
            logger.debug(
                f"Rectangular domain requested. Adjusting nelx to {nelx:d} cells to better match aspect ratio.")
        metamaterial.mesh = fe.RectangleMesh(P0, P1, nelx, nely, 'crossed')
        metamaterial.domain_shape = domain_shape
    elif 'quad' in mesh_cell_type:
        logger.debug("Creating mesh with quadrilateral cells")
        metamaterial.mesh = fe.RectangleMesh.create([fe.Point(0, 0), fe.Point(1, 1)],
                                                    [nelx, nely],
                                                    fe.CellType.Type.quadrilateral)
    else:
        raise ValueError(f"Invalid cell_type: {mesh_cell_type}")
    metamaterial.create_function_spaces(elem_degree=elem_degree)
    metamaterial.initialize_variational_forms()
    return metamaterial


class Metamaterial:
    def __init__(self, E_max: float, E_min: float, nu: float, nelx: int, nely: int, mesh: fe.Mesh = None, x: fe.Function = None, domain_shape: str = None, profile: bool = False):
        self.prop: Properties = Properties(E_max, E_min, nu)
        self.nelx: int = nelx
        self.nely: int = nely
        self.x: fe.Function = x
        self.mesh: fe.Mesh = mesh
        self.domain_shape: str = domain_shape
        self.mirror_map = None

        self._run_times = []
        self.enable_profiling: bool = profile

        self.solver: fe.LUSolver = fe.LUSolver()

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

    def plot_density(self, ax=None, cmap='gray', block=True):
        r = fe.Function(self.R)
        r.vector()[:] = self.x.vector()[:]
        if cmap == 'gray':
            r.vector()[:] = 1. - r.vector()[:]
        r.set_allow_extrapolation(True)

        if isinstance(ax, plt.Axes):
            plt.sca(ax)
        else:
            fig, ax = plt.subplots()
        ax.clear()
        ax.margins(x=0, y=0)
        title = f"Density - Average {np.mean(self.x.vector()[:]):.3f}"
        ax.set_title(title)

        cell_type = self.R.ufl_cell().cellname()
        if cell_type == 'quadrilateral':
            logger.debug("Plotting image using quad cells")
            x_vec = r.vector()[:]
            nely = np.sqrt(x_vec.size).astype(int)
            nelx = nely
            plt.imshow(x_vec.reshape((nely, nelx)),
                       cmap=cmap, vmin=0, vmax=1)
            return

        fe.plot(r, cmap=cmap, vmin=0, vmax=1, title=title)
        if block:
            plt.show(block=block)

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
        R_cg = fe.FunctionSpace(self.mesh, 'CG', 1, constrained_domain=PBC)
        R_grad = fe.VectorFunctionSpace(
            self.mesh, 'CG', 1, constrained_domain=PBC)
        R_tri = fe.FunctionSpace(fe.UnitSquareMesh(
            self.nelx, self.nely, 'crossed'), 'DG', 0)

        self.x = fe.Function(R)
        self.PBC = PBC
        self.W = W
        self.V = W.sub(0).collapse()
        self.R, self.R_cg, self.R_grad = R, R_cg, R_grad
        self.R_tri = R_tri

    def initialize_variational_forms(self):
        self.v_, self.lamb_ = fe.TestFunctions(self.W)
        self.dv, self.dlamb = fe.TrialFunctions(self.W)

        self.E = fe.Function(self.x.function_space())

        self.m_strain = fe.Constant(((0., 0.), (0., 0.)))

        self.a_form = fe.inner(
            linear_stress(linear_strain(self.dv), self.E, self.prop.nu),
            linear_strain(self.v_))*fe.dx
        self.a_form += fe.dot(self.lamb_, self.dv)*fe.dx + \
            fe.dot(self.dlamb, self.v_)*fe.dx

        self.L_form = -fe.inner(
            linear_stress(self.m_strain, self.E, self.prop.nu),
            linear_strain(self.v_))*fe.dx

    def _project_uChom_to_matrix(self, uChom):
        projected_values = np.empty((9, self.R.dim()))  # Preallocate the array

        for idx, (i, j) in enumerate(((i, j) for i in range(3) for j in range(3))):
            projected_function = fe.project(uChom[i][j], self.R)
            projected_values[idx, :] = projected_function.vector().get_local()

        return projected_values

    def homogenized_C(self, sols, E, nu):
        s_list = [linear_stress(linear_strain(u) + macro_strain(i), E, nu)
                  for i, u in enumerate(sols)]

        uChom = [
            [
                fe.inner(s_t, linear_strain(u) + macro_strain(j))
                for j, u, in enumerate(sols)
            ]
            for s_t in s_list
        ]

        uChom_matrix = self._project_uChom_to_matrix(
            uChom) * self.cell_vol / self.domain_volume
        Chom = np.reshape(np.sum(uChom_matrix, axis=1), (3, 3))

        return Chom, uChom_matrix

    def get_dChom(self, sols):
        E_max, E_min, nu = self.prop.E_max, self.prop.E_min, self.prop.nu
        return self.homogenized_C(sols, E_max - E_min, nu)[1]

    def solve(self, debug=False):
        with profile_fem_solution(enabled=self.enable_profiling):
            return self._solve_impl(debug)

    def _solve_impl(self, debug=False):
        """Implementation of the solve method that can be called with or without profiling"""
        if debug:
            self.plot_density()
        self.E.vector()[:] = self.prop.E_min + \
            (self.prop.E_max - self.prop.E_min) * self.x.vector()[:]

        A = fe.assemble(self.a_form)

        # Set the matrix for our reusable solver
        self.solver.set_operator(A)

        sols = []
        for (j, case) in enumerate(["Exx", "Eyy", "Exy"]):
            w = fe.Function(self.W)
            self.m_strain.assign(macro_strain(j))

            b = fe.assemble(self.L_form)

            with profile_solve(enabled=self.enable_profiling):
                self.solver.solve(w.vector(), b)

            v = fe.split(w)[0]
            sols.append(v)

        Chom = self.homogenized_C(sols, self.E, self.prop.nu)[0]

        return sols, Chom

    def get_profiling_report(self):
        """Generate a profiling report summary"""
        if not hasattr(self, 'enable_profiling') or not self.enable_profiling:
            return "Profiling is disabled"

        # Generate FEM profiler report
        fem_profiler.calculate_metrics()
        fem_profiler.report()

        return "Profiling report generated"

    @cached_property
    def cell_vol(self):
        return next(fe.cells(self.mesh)).volume()

    @cached_property
    def resolution(self):
        x_min, y_min = self.mesh.coordinates().min(axis=0)
        x_max, y_max = self.mesh.coordinates().max(axis=0)
        return (x_max - x_min) / self.nelx, (y_max - y_min) / self.nely

    @cached_property
    def domain_volume(self):
        return fe.assemble(fe.Constant(1)*fe.dx(domain=self.mesh))

    @cached_property
    def volume_fraction(self):
        return fe.assemble(self.x * fe.dx)

    @cached_property
    def width(self):
        return self.resolution[0] * self.nelx

    @cached_property
    def height(self):
        return self.resolution[1] * self.nely

    @cached_property
    def cell_midpoints(self):
        return np.array([c.midpoint().array()[:2] for c in fe.cells(self.mesh)])

    def check_gradient(self):
        from tqdm import tqdm

        logger.warning("Running finite difference checker overwrites self.x")
        np.random.seed(0)
        init_x = np.random.uniform(0., 1., size=self.R.dim())

        # ===== ANALYTICAL GRADIENT =====
        self.x.vector()[:] = init_x
        grad_analytical = self.get_dChom(self._solve_impl()[0])

        # ===== FD GRADIENT =====
        grad_fd = np.zeros((9, init_x.size))

        eps = 1e-6
        for i in tqdm(range(init_x.size), desc="Checking Metamaterial gradient"):
            x_plus = init_x.copy()
            x_minus = init_x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps

            self.x.vector()[:] = x_plus
            Chom_plus = self._solve_impl()[1]
            self.x.vector()[:] = x_minus
            Chom_minus = self._solve_impl()[1]

            grad_fd[:, i] = ((Chom_plus - Chom_minus) / (2 * eps)).flatten()

        try:
            npt.assert_allclose(grad_fd, grad_analytical, rtol=1e-5, atol=1e-8)
            logger.info(
                "PASS: Metamaterial gradient check")
        except AssertionError as e:
            logger.error(f"Metamaterial gradient check failed")
            logger.error(
                f"E_max: {self.prop.E_max:.3e}, E_min: {self.prop.E_min:.3e}, nu: {self.prop.nu:.3e}")
            logger.error(e)
            raise e
        except Exception as e:
            logger.error(
                f"Unexpected error occured running metamaterial gradient check: {e}")
            raise e


@dataclass
class Properties:
    E_max: float
    E_min: float
    nu: float
    K: float = field(init=False)
    ambda_: float = field(init=False)
    mu_: float = field(init=False)

    def __post_init__(self):
        self.lambda_, self.mu_ = lame_parameters(
            self.E_max, self.nu, model='plane_stress')
        self.K = self.lambda_ + 2.0*self.mu_
