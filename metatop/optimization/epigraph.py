import abc
import inspect

import fenics as fe
import jax
import jax.interpreters
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nlopt
import numpy as np
from jax.numpy.linalg import inv as jinv
from loguru import logger
from matplotlib import gridspec
from nlopt import ForcedStop

from metatop.filters import jax_projection
from metatop.image import bitmapify
from metatop.mechanics import inv_mandelize, mandelize, ray_q
from metatop.optimization import OptimizationState
from metatop.optimization.utils import (ScalarOptimizationComponent,
                                        VectorOptimizationComponent,
                                        stop_on_nan)
from metatop.profiling import profile_block, profile_function


def spec_norm(x: np.ndarray):
    return jnp.linalg.norm(x, ord=2)


class EpigraphOptimizer(nlopt.opt):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.active_constraints = []

    def optimize(self, x):
        x[-1] = self._update_t(x)
        logger.info(f"New t value: {x[-1]:.3e}")
        return super().optimize(x)

    def _update_t(self, x):
        logger.info(f"Updating t...")
        logger.info(f"Old t value {x[-1]:.3e}")
        x[-1] = 0.
        new_t = -np.inf
        for g in self.active_constraints:
            if isinstance(g, EpigraphComponent):
                logger.info(f"Accounting for constraint {g} in t update")
                if isinstance(g, VectorOptimizationComponent):
                    results = np.zeros(g.n_constraints)
                    g(results, x, np.array([]), dummy_run=True)
                    print('results', results)
                    new_t = max(new_t, *(results))
                elif isinstance(g, ScalarOptimizationComponent):
                    c = g(x, np.array([]))
                    new_t = max(new_t, c)
                else:
                    logger.warning(
                        f"Cannot account for constraint {g} in update t.")
            else:
                logger.info(f"Skipping constraint {g} in t update")
        return new_t

    def add_inequality_mconstraint(self, *args, uses_t=True):
        con = args[0]
        if isinstance(con, EpigraphComponent):
            self.active_constraints.append(args[0])
        else:
            raise ValueError("First argument must be a constraint")
        return super().add_inequality_mconstraint(*args)

    def add_inequality_constraint(self, *args):
        con = args[0]
        if isinstance(con, EpigraphComponent):
            self.active_constraints.append(args[0])
        else:
            raise ValueError("First argument must be a constraint")
        return super().add_inequality_constraint(*args)

    def add_equality_constraint(self, *args):
        raise NotImplementedError(
            "Equality constraints are not supported for this optimizer")

    def add_equality_mconstraint(self, *args):
        raise NotImplementedError(
            "Equality constraints are not supported for this optimizer")

    @property
    def size(self):
        return self.get_dimension()

    @property
    def n_constraints(self):
        return len(self.active_constraints)


class EpigraphComponent:
    """Marker interface for constraints that are bound in some way by the auxilary t variable"""

    def split_t(self, x):
        return x[:-1], x[-1]


class EpigraphObjective(ScalarOptimizationComponent, EpigraphComponent):
    """
    A class representing a minimax optimization problem.
    We reformulate the problem as a nonlinear optimization problem by adding a slack variable t as the objective function.

    min_x max{f1, f2, f3}
    s.t. g(x) <= 0

    becomes

    min_{x, t} t
    s.t. f1 <= t
         f2 <= t
         f3 <= t
         g(x) <= 0

    The objective function is simply the slack variable t.
    The gradiant of the objective function is then all zeros with a 1 at the end of the vector.
    """

    def __call__(self, x, grad):
        """
        Evaluates the objective function of the minimax problem.

        Parameters:
        - x: The input vector.
        - grad: The gradient vector.

        Returns:
        - The objective function value.
        """
        _, t = self.split_t(x)

        if grad.size > 0:
            grad[:-1], grad[-1] = 0., 1.

        if t > 1e10:
            raise ForcedStop(
                "Objective function value is too large. Terminating optimization run.")

        if not self.silent:
            self.ops.obj_n_calls += 1
            logger.info(f"{self.ops.obj_n_calls}:")
            logger.info(f"{self.__str__()} t={t:.4f}")
            self.ops.update_evals(self.__str__(), t)
            self.ops.update_plot(self.__str__(), labels=['t'])

        stop_on_nan(t)
        return t

    def eval(self):
        return

    def adjoint(eval):
        return


class PrimaryEpigraphConstraint(VectorOptimizationComponent, EpigraphComponent):

    def __init__(self, ops: OptimizationState, objective_type: str, eps: float = 0., verbose: bool = False, silent: bool = False):
        super().__init__(ops, eps=eps, verbose=verbose, silent=silent)
        self.obj_type = objective_type

        self._available_obj_types = ('ray', 'ratio')
        if self.obj_type not in self._available_obj_types:
            raise ValueError(
                f"Objective type {self.obj_type} is not available. Types are [{self._available_obj_types}]")

    def __call__(self, results, x, grad, dummy_run=False):
        x_, t = self.split_t(x)
        sols, Chom, dChom_dxfem, dxfem_dx_vjp = self.forward(x_)

        c, cs = self.eval(Chom)
        results[:] = c - t

        if not self.silent:
            logger.info(f"{self.__str__()}")
            logger.info(f"g(x): {c}")
            logger.info(f"Raw Rayleigh Quotients: {cs}")

        if dummy_run:
            return

        dc_dChom = jax.jacrev(self.eval,
                              has_aux=True)(Chom)[0].reshape((self.n_constraints, 9))

        if grad.size > 0:
            grad[:] = self.adjoint(dc_dChom, dChom_dxfem, dxfem_dx_vjp)

        self.ops.update_state(sols, Chom, dChom_dxfem,
                              dxfem_dx_vjp, x_, increment_obj_n_calls=False)
        id = self.__str__()
        self.ops.update_evals(id, c,)
        A = 'C' if self.ops.extremal_mode == 1 else 'S'
        self.ops.update_plot(id, is_primary=True, labels=[
                             rf"$R({A},v_1)$", rf"$1-R({A},v_2)$", rf"$1-R({A},v_3)$"])

    def eval(self, C: jnp.ndarray):
        M = mandelize(C)
        M = jnp.linalg.inv(M) if self.ops.extremal_mode == 2 else M
        M /= spec_norm(M, ord=2)

        V = self.ops.basis_v
        r1, r2, r3 = ray_q(M, V)

        if self.obj_type == 'ray':
            return jnp.log(jnp.array([r1, 1-r2, 1-r3])), jnp.array([r1, r2, r3])
        elif self.obj_type == 'ratio':
            return jnp.log(jnp.array([r1/r2, r1/r3])), jnp.array([r1, r2, r3])
#         elif self.objective_type == 'ray_sq':
#             return (jnp.log(jnp.array([r1**2, (1. - r2**2), (1. - r3**2), ])+1e-8), jnp.array([r1, r2, r3, ]))
#         elif self.objective_type == 'uni':
#             return (jnp.log(jnp.array[r1, r2 - r1 + 1e-3, r3 - r1 + 1e-3]), jnp.array([r1, r2, r3]))
#         # elif self.objective_type == 'norm':
#             # return jnp.log(self.w*(jnp.array([jnorm(Cv1), 1-jnorm(Cv2), 1-jnorm(Cv3)])+1e-8)), jnp.array([jnorm(Cv1), jnorm(Cv2), jnorm(Cv3)])
#         # elif self.objective_type == 'norm_sq':
#             # return (jnp.log(self.w*jnp.array([Cv1@Cv1, 1 - Cv2@Cv2, 1-Cv3@Cv3, ])+1e-8), jnp.array([Cv1@Cv1, Cv2@Cv2, Cv3@Cv3]))
#         elif self.objective_type == 'ratio_sq':
#             return jnp.log(jnp.array([(r1/r2)**2, (r1/r3)**2, ])), jnp.array([r1, r2, r3])
#         elif self.objective_type == 'ratio_c1sq':
#             return jnp.log(jnp.array([r1**2/r2, r1**2/r3, ])), jnp.array([r1, r2, r3])
#         else:
#             raise ValueError(
#                 f"Objective '{self.objective_type}' type not found.")

    def adjoint(self, dc_dChom, dChom_dxfem, dxfem_dx_vjp):
        nc = self.n_constraints
        dc_dx = np.zeros((nc, dChom_dxfem.shape[1]+1))
        for n in range(nc):
            dc_dx[n, :-1] = dxfem_dx_vjp(dc_dChom[n, :] @ dChom_dxfem)[0]
            dc_dx[n, -1] = -1.
        return dc_dx

    @property
    def n_constraints(self):
        return 2 if 'ratio' in self.obj_type else 3


# class EigenvalueProblemConstraints(EpigraphConstraint):
#     def __init__(self, basis_v, ops, metamaterial, extremal_mode, weights=jnp.ones(3), check_valid=False, **kwargs):
#         # Initialize with base EpigraphConstraint initialization
#         super().__init__(basis_v, ops, metamaterial, extremal_mode, **kwargs)

#         self.weights = weights
#         self.check_valid = check_valid
#         self.n_constraints = 7

#         # Setup plots same as ExtremalConstraints
#         self.img_resolution = (200, 200)
#         self.img_shape = (self.metamaterial.width, self.metamaterial.height)

#     def obj(self, x_: np.ndarray, C):
#         # We grab U out of x_ so ease derivative computation
#         return self._obj_wrt_U(self._select_U(x_), C)

#     # We only care about the derivative of the objective w.r.t. the specific parts of x that comprise U.
#     # So we do this subcall to avoid differentiating against all of x.
#     def _obj_wrt_U(self, U, C):
#         M = mandelize(C)
#         M = jnp.linalg.inv(M) if self.extremal_mode == 2 else M
#         M /= jnorm(M, ord=2)

#         # Rayleigh quotients with U
#         r1, r2, r3 = self.weights*ray_q(M, U)
#         rays = jnp.array([r1, 1.-r2, 1.-r3])

#         V = self.basis_v
#         U_norms = jnp.linalg.norm(U, axis=0)
#         V_norms = jnp.linalg.norm(V, axis=0)
#         cosines = (1 + 1e-6 - jnp.diag(U.T@V) / (U_norms*V_norms))/self.eps

#         ortho = jnp.linalg.norm(U.T @ U - jnp.eye(3), ord='fro')

#         return (jnp.log(jnp.hstack([rays, cosines, [ortho]])),
#                 jnp.hstack([jnp.array([r1, r2, r3]), cosines, [ortho]]))

#     def adjoint(self, x_, grad, dxfem_dx_vjp, Chom, dChom_dxfem):
#         # argnums=[0,1] because we care about both derivatives
#         jac_func = jax.jacrev(self._obj_wrt_U, argnums=[0, 1], has_aux=True)
#         dc_dU, dc_dChom = jac_func(self._select_U(x_), Chom)[0]

#         # both arguments get passed in a 3x3 matrices so we unfold them for easier referencing
#         dc_dU = dc_dU.reshape((self.n_constraints, 9))
#         dc_dChom = dc_dChom.reshape((self.n_constraints, 9))

#         for n in range(self.n_constraints):
#             grad[n, :-10] = dxfem_dx_vjp(dc_dChom[n, :] @ dChom_dxfem)[0]
#             grad[n, -10:-1] = dc_dU[n, :]
#             grad[n, -1] = -1.

#     def forward(self, x_):
#         return super().forward(self._strip_U(x_))

#     def update_plot(self, x_):
#         super().update_plot(self._strip_U(x_))

#     def _strip_U(self, x_):
#         return x_[:-9]

#     def _select_U(self, x_):
#         return x_[-9:].reshape((3, 3))

#     def _check_validity(self, x_):
#         if not self.check_valid:
#             return
#         if not self._strip_U(x_).size == self.metamaterial.R.dim():
#             raise ValueError(f"Mismatching size of x and R.dim()")

#     def _setup_evals_lines(self):
#         """Setup evaluation lines specific to EigenvalueProblemConstraints"""
#         d = np.ones(self.n_constraints+1)
#         labels = ['$t$',
#                   '$u_1^TCu_1$',
#                   '$1-u_2^TCu_2$',
#                   '$1-u_3^TCu_3$',
#                   '$\cos(u_1,v_1)$',
#                   '$\cos(u_2,v_2)$',
#                   '$\cos(u_3,v_3)$',
#                   'ortho']
#         self.evals_lines = self.ax2.plot(
#             [d], [d], marker='.',
#             label=labels
#         )
#         self.ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

#     def __str__(self):
#         return "EigenvalueProblemConstraints"


# class SpectralNormConstraint:

#     def __init__(self, ops, bound=1., verbose=True):
#         self.ops = ops
#         self.bound = bound
#         self.verbose = verbose

#     def __call__(self, x, grad, ):

#         Chom, dChom_dxfem, dxfem_dx_vjp = self.ops.Chom, self.ops.dChom_dxfem, self.ops.dxfem_dx_vjp

#         m = jnp.diag(np.array([1., 1., np.sqrt(2)]))

#         def g(C):
#             C = m @ C @ m
#             return jnp.linalg.norm(C, ord=2)
#             # return jnp.trace(C)

#         c, dc_dChom = jax.value_and_grad(g)(jnp.asarray(Chom))

#         if grad.size > 0:
#             grad[:-1] = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]
#             grad[-1] = 0.

#         if self.verbose:
#             print(f"Spectral Norm Constraint:")
#             print(f"Value: {c:.3f} (Target >={self.bound:.3f})")
#             # print(f"Eigenvalues: {np.linalg.eigvalsh(m@Chom@m)}")
#         return float(self.bound - c)

#     def __str__(self):
#         return "SpectralNormConstraint"


# class EigenvectorConstraint:

#     def __init__(self, basis_v, ops, eps=1e-3, verbose=True):
#         self.basis_v = basis_v
#         self.ops = ops
#         self.eps = eps
#         self.verbose = verbose

#         self.n_constraints = 3

#     def __call__(self, results, x, grad, dummy_run=False):

#         Chom, dChom_dxfem, dxfem_dx_vjp = self.ops.Chom, self.ops.dChom_dxfem, self.ops.dxfem_dx_vjp

#         t = x[-1]

#         c, cs = self.obj(Chom)
#         stop_on_nan(c)
#         results[:] = c - t

#         if dummy_run:
#             return

#         if grad.size > 0:
#             dc_dChom = jax.jacrev(self.obj, has_aux=True)(jnp.asarray(Chom))[
#                 0].reshape((self.n_constraints, 9))
#             for n in range(self.n_constraints):
#                 grad[n, :-1] = dxfem_dx_vjp(dc_dChom[n, :] @ dChom_dxfem)[0]
#                 grad[n, -1] = -1.
#                 # grad[n,:] = dxfem_dx_vjp(dc_dChom[n,:] @ dChom_dxfem)[0]

#         if self.verbose:
#             print(f"Eigenvector Constraint:")
#             print(f"Values: {c}")
#             print(f"Residuals: {cs}")
#         else:
#             print(f"\tg_vec(x): {c}")

#     def obj(self, C):
#         M = mandelize(C)
#         # eigenvectors are the same for C and S so we don't worry about inverting like we do in other constraints
#         M /= jnorm(M, ord=2)

#         V = self.basis_v
#         # Rayleigh quotient in diagonal matrix form
#         R = jnp.diag(ray_q(M, V))
#         # Resdiuals of eigenvector alignment
#         res = M @ V - V @ R
#         # norm squared of each residual
#         norm_sq = jnp.sum(jnp.square(res), axis=0)

#         return jnp.log(norm_sq/self.eps), norm_sq

#     def __str__(self):
#         return "EigenvectorConstraint"


# class TraceConstraint:

#     def __init__(self, ops, bound=3e-1, verbose=True):
#         self.ops = ops
#         self.verbose = verbose
#         self.bound = bound

#     def __call__(self, x, grad):

#         Chom, dChom_dxfem, dxfem_dx_vjp = self.ops.Chom, self.ops.dChom_dxfem, self.ops.dxfem_dx_vjp

#         m = jnp.diag(np.array([1., 1., np.sqrt(2)]))

#         def obj(C):
#             return -jnp.trace(m@C@m)

#         c, dc_dChom = jax.value_and_grad(obj)(Chom)

#         if grad.size > 0:
#             grad[:-1] = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]
#             grad[-1] = 0.

#         print(f"\tg_trc(x): {-c:.3f} (Target >={self.bound:.3f})")

#         return float(self.bound + c)

#     def __str__(self):
#         return "TraceConstraint"


# class InvariantsConstraint:

#     def __init__(self, ops, verbose=True):
#         self.ops = ops
#         self.verbose = verbose
#         # Invariant bounds:
#         # tr(C) >= eps --> eps - tr(C) <= 0 --> (-tr(C)) - (-eps) <= 0
#         self.eps = np.array([-3e-1, -0., 1e-1])

#         self.n_constraints = 3

#         assert self.eps.size == self.n_constraints, "Epsilons must be the same length as the number of constraints"

#     def __call__(self, results, x, grad, dummy_run=False):

#         Chom, dChom_dxfem, dxfem_dx_vjp = self.ops.Chom, self.ops.dChom_dxfem, self.ops.dxfem_dx_vjp

#         t = x[-1]

#         def obj(C):
#             m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
#             C = m @ C @ m
#             I1 = jnp.trace(C)
#             I2 = 0.5 * (jnp.trace(C)**2 - jnp.trace(C @ C))
#             I3 = jnp.linalg.det(C)
#             return jnp.array([-I1, -I2, I3])

#         c = obj(jnp.asarray(Chom))
#         results[:] = c - self.eps

#         if dummy_run:
#             return

#         if grad.size > 0:
#             dc_dChom = jax.jacrev(obj)(jnp.asarray(
#                 Chom)).reshape((self.n_constraints, 9))
#             for n in range(self.n_constraints):
#                 grad[n, :-1] = dxfem_dx_vjp(dc_dChom[n, :] @ dChom_dxfem)[0]
#                 grad[n, -1] = 0.

#         if self.verbose:
#             print(f"Invariant Constraint:")
#             print(f"Trace: {-c[0]:.3f} (Target >={-self.eps[0]:.3f})")
#             print(
#                 f"Second Invariant: {-c[1]:.2e} (Target >={-self.eps[1]:.3f})")
#             print(f"Det: {c[2]:.2e} (Target <={self.eps[2]:.3f})")


# class GeometricConstraints:

#     def __init__(self, ops, metamaterial, line_width, line_space, c, eps=1e-3, verbose=True):
#         self.ops = ops
#         self.metamaterial = metamaterial
#         # if 'quad' not in metamaterial.mesh.ufl_cell().cellname():
#         # raise ValueError("Geometric Constraints only work with quadrilateral elements")
#         self.lw = line_width
#         self.ls = line_space
#         self.filt_radius = self.ops.filt.radius
#         self.eps = eps
#         self.verbose = verbose
#         self.c = c
#         self.n_constraints = 2

#         # items to help calculate the gradient of rho_tilde
#         self._r_tilde = fe.Function(self.metamaterial.R)

#     def __call__(self, results, x, grad, dummy_run=False):

#         filt_fn = self.ops.filt_fn
#         x, t = x[:-1], x[-1]

#         def g(x):
#             x_tilde = filt_fn(x)
#             a1 = jnp.minimum(x_tilde - self._eta_e, 0.)**2
#             b1 = self._indicator_fn(x, 'width')
#             f, a = plt.subplots(1, 3)
#             plt.sca(a[0])
#             plt.imshow(a1.reshape((100, 100)))
#             plt.colorbar()
#             plt.sca(a[1])
#             plt.imshow(b1.reshape((100, 100)))
#             plt.colorbar()
#             plt.sca(a[2])
#             plt.imshow((a1*b1).reshape((100, 100)))
#             plt.colorbar()
#             c1 = jnp.mean(a1*b1)

#             a2 = jnp.minimum(self._eta_d - x_tilde, 0.)**2
#             b2 = self._indicator_fn(x, 'space')
#             c2 = jnp.mean(a2*b2)

#             return jnp.log(jnp.array([c1, c2]))

#         c = g(x)
#         dc_dx = jax.jacrev(g)(x)

#         results[:] = c - t*self.eps

#         if grad.size > 0:
#             for n in range(self.n_constraints):
#                 grad[n, :-1] = dc_dx[n, :]
#                 grad[n, -1] = -self.eps

#         if self.verbose:
#             print(f"Geometric Constraint:")
#             print(f"Width: {c[0]:.3e} (Target ≤{t*self.eps:.1e})")
#             print(f"Space: {c[1]:.3e} (Target ≤{t*self.eps:.1e})")

#     def _indicator_fn(self, x, type):
#         if type not in ['width', 'space']:
#             raise ValueError("Indicator Function must be 'width' or 'space'")
#         filt_fn, beta, eta = self.ops.filt_fn, self.ops.beta, self.ops.eta
#         nelx, nely = self.metamaterial.nelx, self.metamaterial.nely

#         x_tilde = filt_fn(x)
#         x_bar = jax_projection(x_tilde, beta, eta)

#         r = fe.Function(self.metamaterial.R)
#         r.vector()[:] = x
#         # here we use fenics gradient to calculate grad(rho_tilde)
#         # first we convert x_tilde the vector to a fenics function in DG space
#         self._r_tilde.vector()[:] = x_tilde
#         # then we project r_tilde to a CG space so that we can get the gradient
#         # this is required because the gradient of a DG function is zero (values are constant across the cell)
#         r_tilde_cg = fe.project(self._r_tilde, self.metamaterial.R_cg)
#         # now we can calculate the inner product of the gradient of r_tilde and project back to the original DG space
#         grad_r_tilde = fe.grad(r_tilde_cg)
#         laplac_r_tilde = fe.div(grad_r_tilde)
#         grad_r_tilde_norm_sq = fe.project(
#             fe.inner(grad_r_tilde, grad_r_tilde), self.metamaterial.R).vector()[:].reshape((nely, nelx))

#         r_tilde_img = x_tilde.reshape((nely, nelx))

#         def fd_norm_sq(img):
#             grad = self._fd_grad(img, h=1 / nelx)
#             return grad[0]**2 + grad[1]**2
#         fd_grad_r_tilde_norm_sq = fd_norm_sq(r_tilde_img)
#         J_fd = jax.jacfwd(fd_norm_sq)(r_tilde_img)
#         J_fenics = -2*fe.div(grad_r_tilde)
#         # grad_rho_img = self._fd_grad(r_tilde_img, h=1 / nelx)
#         # fd_grad_r_tilde_norm_sq = (grad_rho_img[1]**2 + grad_rho_img[0]**2).reshape((nely, nelx))
#         # d_check = jax.jacrev(self._fd_grad)(r_tilde_img)
#         # d_check = jax.gradient(self._fd_grad)(r_tilde_img)
#         # d_check = jax.jacrev(jnp.gradient)(r_tilde_img)

#         # fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
#         # plt.sca(ax0)
#         # plt.imshow(grad_r_tilde_norm_sq.vector()[
#         #            :].reshape((nely, nelx)), cmap='gray')
#         # plt.colorbar()
#         # plt.title("grad_r_tilde_norm_sq")

#         # plt.sca(ax1)
#         # plt.imshow(fd_grad_r_tilde_norm_sq, cmap='gray')
#         # plt.colorbar()
#         # plt.title("fd_grad_r_tilde_norm_sq")

#         # plt.sca(ax2)
#         # # plt.imshow((grad_x_tilde_norm_sq - grad_r_tilde_norm_sq.vector()[:].reshape((nely, nelx))))
#         # diff = fd_grad_r_tilde_norm_sq.flatten(
#         # ) - grad_r_tilde_norm_sq.vector()[:]
#         # err = fe.Function(self.metamaterial.R)
#         # err.vector()[:] = diff
#         # plt.plot(diff)
#         # # plt.yscale('log')
#         # # plt.colorbar()
#         # plt.title("diff")
#         # plt.show()
#         # print(f"Max diff: {np.max(diff)}")
#         # print(f"Min diff: {np.min(diff)}")
#         # print(f"Mean diff: {np.mean(diff)}")
#         # print(f"Std diff: {np.std(diff)}")
#         # print(f"Norm diff: {np.linalg.norm(diff)/nelx}")
#         # print(f"Fenics norm diff: {norm(err, 'L2')}")

#         q = jnp.exp(-self.c * (grad_r_tilde_norm_sq))
#         fig, (ax0, ax1) = plt.subplots(1, 2)
#         plt.sca(ax0)
#         plt.imshow(grad_r_tilde_norm_sq, cmap='gray')
#         plt.show()

#         if type == 'width':
#             return x_bar * q.flatten()
#         elif type == 'space':
#             return (1. - x_bar) * q.flatten()
#         else:
#             raise ValueError("Indicator Function must be 'width' or 'space'")

#     def _calculate_etas(self, lw, ls, R):
#         eta_e, eta_d = 1., 0.
#         lwR, lsR = lw/R, ls/R

#         if lwR < 0.:
#             raise ValueError("Line width / Radius must be greater than 0.")
#         elif 0 <= lwR < 1.:
#             eta_e = 0.25*lwR**2 + 0.5
#         elif 1. <= lwR <= 2.:
#             eta_e = -0.25*lwR**2 + lwR

#         if lsR < 0.:
#             raise ValueError("Line space / Radius must be greater than 0.")
#         elif 0 <= lsR < 1.:
#             eta_d = 0.5 - 0.25*lsR**2
#         elif 1. <= lsR <= 2.:
#             eta_d = 1. + 0.25*lsR**2 - lsR

#         return eta_e, eta_d

#     def _fd_grad(self, img, h=None):
#         h = self.metamaterial.resolution[0] if h is None else h
#         if self.ops.filt.distance_method == 'periodic':
#             # use jnp.roll instead of jnp.gradient b/c periodic boundary conditions
#             # right_neighbors  = jnp.roll(img, -1, axis=1)
#             # left_neighbors   = jnp.roll(img, 1, axis=1)
#             # top_neighbors    = jnp.roll(img, -1, axis=0)
#             # bottom_neighbors = jnp.roll(img, 1, axis=0)

#             # grad_x = (right_neighbors - left_neighbors) / 2. / h
#             # grad_y = (top_neighbors - bottom_neighbors) / 2. / h

#             # Compute neighbors using periodic boundary conditions with jnp.roll
#             right1 = jnp.roll(img, -1, axis=1)
#             left1 = jnp.roll(img, 1, axis=1)
#             right2 = jnp.roll(img, -2, axis=1)
#             left2 = jnp.roll(img, 2, axis=1)

#             top1 = jnp.roll(img, -1, axis=0)
#             bottom1 = jnp.roll(img, 1, axis=0)
#             top2 = jnp.roll(img, -2, axis=0)
#             bottom2 = jnp.roll(img, 2, axis=0)

#             # Compute fourth-order central differences
#             grad_x = (-right2 + 8*right1 - 8*left1 + left2) / (12 * h)
#             grad_y = (-top2 + 8*top1 - 8*bottom1 + bottom2) / (12 * h)

#         else:  # assume non-periodicity
#             grad_y, grad_x = jnp.gradient(img, h)

#         # match return format of jnp.gradient
#         return grad_y, grad_x


# class OffDiagonalConstraint:
#     def __init__(self, v, ops, eps=1e-3, verbose=True):
#         self.v = v
#         self.ops = ops
#         self.eps = eps
#         self.verbose = verbose

#         self.n_constraints = 1

#     def __call__(self, x, grad):

#         Chom, dChom_dxfem, dxfem_dx_vjp = self.ops.Chom, self.ops.dChom_dxfem, self.ops.dxfem_dx_vjp

#         c, dc_dChom = jax.value_and_grad(self.obj)(Chom)

#         if grad.size > 0:
#             grad[:-1] = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]

#         print(f"\tg_dia(x): {c}")

#         return float(c)

#     def obj(self, C):
#         m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
#         C = m @ C @ m
#         vCv = self.v.T @ C @ self.v
#         return jnp.log((vCv[0, 1]**2 + vCv[0, 2]**2 + vCv[1, 2]**2)/self.eps)
#         # return jnp.linalg.norm(vCv - jnp.diag(jnp.diag(vCv)))

#     def __str__(self):
#         return "OffDiagonalConstraint"


# class VolumeConstraint:

#     def __init__(self, ops, bound, verbose=True):
#         self.ops = ops
#         self.bound = bound
#         self.verbose = verbose

#     def __call__(self, x, grad):

#         x_fem, dxfem_dx_vjp = self.ops.x_fem, self.ops.dxfem_dx_vjp

#         g, dg_dx_fem = jax.value_and_grad(jnp.mean)(x_fem)

#         if grad.size > 0:
#             grad[:-1] = dxfem_dx_vjp(dg_dx_fem)[0]
#             grad[-1] = 0.

#         if self.verbose:
#             print(f"Volume: {g:.3f} (Target <= {self.bound:.3f})")

#         return float(g - self.bound)

#     def __str__(self):
#         return "VolumeConstraint"
