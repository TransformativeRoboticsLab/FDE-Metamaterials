import itertools

import fenics as fe
import jax
import jax.interpreters
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.numpy.linalg import norm as jnorm
from matplotlib import gridspec
from nlopt import ForcedStop

from metatop.filters import jax_projection
from metatop.image import bitmapify
from metatop.mechanics import mandelize, ray_q
from metatop.optimization import OptimizationState

from .OptimizationComponents import *
from .utils import stop_on_nan


class ScalarObjective(ScalarOptimizationComponent):

    def __call__(self, x, grad):

        sols, Chom, dChom_dxfem, dxfem_dx_vjp = self.forward(x)

        try:
            (c, cs), dc_dChom = jax.value_and_grad(
                self.eval, has_aux=True)(Chom)
        except Exception as e:
            logger.error(
                f"Error in calculating value and grad in {self.__str__()}: {e}")
            raise e
        self.ops.update_state(sols, Chom, dChom_dxfem, dxfem_dx_vjp, x)

        if grad.size > 0:
            grad[:] = self.adjoint(dc_dChom, dChom_dxfem, dxfem_dx_vjp)

        id = self.__str__()
        self.ops.update_evals_and_plot(id, c)

        if not self.silent:
            logger.info(f"{self.ops.obj_n_calls}:")
            logger.info(f"{self.__str__()} f(x): {c:.4f}")
            logger.info(f"Raw values:\n{cs}")

        stop_on_nan(c)
        return float(c)


class ScalarConstraint(ScalarOptimizationComponent):

    def __call__(self, _, grad):

        # We don't need to rerun the forward solve so we just bring in the last solved state from the OPS
        Chom, dxfem_dx_vjp, dChom_dxfem = self.ops.Chom, self.ops.dxfem_dx_vjp, self.ops.dChom_dxfem

        # This is a little wasteful if we aren't using gradient based optimizations, but MMA is gradient based so we'll always need it
        (c, cs), dc_dChom = jax.value_and_grad(self.eval, has_aux=True)(Chom)

        if grad.size > 0:
            grad[:] = self.adjoint(dc_dChom, dChom_dxfem, dxfem_dx_vjp)

        id = self.__str__()
        self.ops.update_evals_and_plot(id, cs)

        if not self.silent:
            logger.info(f"{self.__str__()} g(x): {c:2e}")
            logger.info(f"Raw values: {cs}")

        stop_on_nan(c)
        return float(c)


class RayleighMinimaxObjective(ScalarObjective):

    def eval(self, C):
        M = mandelize(C)
        M /= jnorm(M, ord=2)

        V = self.ops.basis_v
        r1, r2, r3 = ray_q(M, V)

        return jnp.max(jnp.array([r1, 1-r2, 1-r3])), jnp.array([r1, r2, r3])


class MatrixMatchingObjective(ScalarObjective):

    def __init__(self, *args, low_val: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        desired_eigenvalues = jnp.diag(jnp.array([low_val, 1., 1.]))

        V = self.ops.basis_v
        self.Mstar = V @ desired_eigenvalues @ V.T
        logger.info(f"Desired M:\n{self.Mstar}")

    def eval(self, C):
        M = mandelize(C)
        M /= jnorm(M, ord=2)
        Mstar = self.Mstar

        diff = M - Mstar
        val = jnp.sum(jnp.square(diff)) * (-1)**(self.ops.extremal_mode+1)
        return val, diff

    def __str__(self):
        return r"$\|M-M^*\|_F$"


class PoissonsRatioObjective(ScalarObjective):

    def eval(self, C):
        M = jnp.linalg.inv(mandelize(C))
        # M /= jnorm(M, ord=2)

        return -M[0, 1] / M[0, 0], M

    def __str__(self):
        return r"$\nu^*$"


class BulkModulusConstraint(ScalarConstraint):

    def __init__(self, *args, a: float = 0.002, **kwargs):
        super().__init__(*args, **kwargs)

        self.eps = a * self._calc_K(self.ops.metamaterial.prop.E_max,
                                    self.ops.metamaterial.prop.nu)
        logger.warning(
            "Ignoring kwarg eps value. BulkModulusConstraint calculates its own based on `a`.")

    def eval(self, C):

        C11 = C[0, 0]
        C12 = C[0, 1]
        C22 = C[1, 1]

        K_plane = (C11 * C22 - C12**2) / C11
        return self.eps - K_plane, K_plane

    def _calc_K(self, E, nu):
        # computes plane stress bulk modulus from E and nu
        K = E / (3 * (1 - 2 * nu))
        G = E / (2 * (1 + nu))
        K_plane = 9.*K*G / (3.*K + 4.*G)
        return K_plane

    def __str__(self):
        return fr"$K^* \geq {self.eps:.2e}$"


@jax.jit
def isotropic_C(C):

    Cii = (C[0, 0] + C[1, 1]) / 2.
    Ckk = (C[0, 0] - C[0, 1]) / 2.
    Ciso = C.at[0, 0].set(Cii)
    Ciso = Ciso.at[1, 1].set(Cii)
    Ciso = Ciso.at[2, 2].set(Ckk)
    return Ciso


class IsotropicConstraint(ScalarConstraint):

    def eval(self, C):

        Ciso = isotropic_C(C)

        diff = C - Ciso
        val = jnp.sum(jnp.square(diff)) / Ciso[0, 0]**2

        return val - self.eps, val

    def __str__(self):
        return r"$\frac{||C-C^{iso}||_F}{(C_{11}^{iso})^2}$"


class RayleighRatioObjective(ScalarObjective):

    def __init__(self, *args, mean_type: str, **kwargs):
        super().__init__(*args, **kwargs)

        if mean_type not in self._eval_fns:
            raise ValueError(
                f"Mean type {mean_type} not in eval_fns.\n" +
                f"Must be one of {self._eval_fns.keys()}")
        self.eval_fn = self._eval_fns[mean_type]

    def eval(self, C):
        M = mandelize(C)
        # M = jnp.linalg.inv(M) if self.extremal_mode == 2 else M
        M /= jnorm(M, ord=2)

        V = self.ops.basis_v
        r1, r2, r3 = ray_q(M, V)
        num = r1
        den = self.eval_fn(jnp.array([r2, r3]))
        return num/den, jnp.array([r1, r2, r3])

    def __str__(self):
        return r"$R(M,v_1) / h(R(M,v_20, R(M,v_3))$"

    @property
    def _eval_fns(self):
        eval_fns = {
            'amean': lambda r: jnp.mean(r),
            'gmean': lambda r: jnp.power(jnp.product(r), 1/r.size),
            'hmean': lambda r: None if jnp.any(r) <= 0 else r.size / jnp.sum(1/r)
        }
        return eval_fns


class EigenvectorConstraint(ScalarConstraint):

    def eval(self, C):
        M = mandelize(C)
        # M = jnp.linalg.norm(M) if self.extremal_mode == 2 else M
        M /= jnorm(M)

        _, e_vecs = jnp.linalg.eigh(M)
        V = self.ops.basis_v
        v1, v2, v3 = V[:, 0], V[:, 1], V[:, 2]
        outer_v1 = jnp.outer(v1, v1)
        outer_v2 = jnp.outer(v2, v2)
        outer_v3 = jnp.outer(v3, v3)

        min_penalty = jnp.inf
        for p in itertools.permutations([0, 1, 2]):
            current_penalty = 0.0
            current_penalty += jnorm(outer_v1 - jnp.outer(
                e_vecs[:, p[0]], e_vecs[:, p[0]]), ord='fro')**2
            current_penalty += jnorm(outer_v2 - jnp.outer(
                e_vecs[:, p[1]], e_vecs[:, p[1]]), ord='fro')**2
            current_penalty += jnorm(outer_v3 - jnp.outer(
                e_vecs[:, p[2]], e_vecs[:, p[2]]), ord='fro')**2
            min_penalty = jnp.minimum(min_penalty, current_penalty)

        return min_penalty - self.eps, min_penalty


class NormEigenvectorConstraint(ScalarConstraint):

    def eval(self, C: jnp.ndarray):
        M = mandelize(C)
        M /= jnorm(M, ord=2)

        V = self.ops.basis_v
        R = jnp.diag(ray_q(M, V))

        res = jnp.sum(jnp.square(M@V - V @ R))

        return res - self.eps, res


class SameLargeValueConstraint(ScalarConstraint):

    def eval(self, C):
        M = mandelize(C)
        # M = jnp.linalg.norm(M) if self.extremal_mode == 2 else M
        M /= jnorm(M, ord=2)

        # e1, e2, e3 = jnp.linalg.eigvalsh(M)
        V = self.ops.basis_v
        r1, r2, r3 = ray_q(M, V)

        out = (r2-r3)**2

        return out - self.eps, out


class TraceObjective(ScalarObjective):

    def eval(self, C):
        M = mandelize(C)

        return -jnp.trace(M)**2, jnp.trace(M)


class TraceConstraint(ScalarConstraint):

    def eval(self, C):
        M = mandelize(C)
        val = jnp.trace(M)

        return self.eps - val, val


class DetObjective(ScalarObjective):

    def eval(self, C):
        M = mandelize(C)

        return jnp.linalg.det(M)**2, jnp.linalg.det(M)


class FirstInvariantConstraint(ScalarConstraint):
    """ The first invariant (trace) must be larger than some prescribed value"""

    def eval(self, C):
        M = mandelize(C)
        I1 = jnp.trace(M)
        return self.eps - I1, I1

    def __str__(self):
        return "tr(M)"


class SecondInvariantConstraint(ScalarConstraint):
    """ The second invariant must be positive """

    def eval(self, C):
        M = mandelize(C)
        I2 = 0.5 * (jnp.trace(M)**2 - jnp.trace(M**2))

        return -I2, I2

    def __str__(self):
        return "I2"


class ThirdInvariantConstraint(ScalarConstraint):
    """ The determinant must be close to zero """

    def eval(self, C):
        M = mandelize(C)
        I3 = jnp.linalg.det(M)
        return I3 - self.eps, I3

    def __str__(self):
        return "det(M)"


class UnimodeErrConstraint(ScalarConstraint):

    def eval(self, C):
        M = mandelize(C)
        v = self.ops.basis_v[:, 0]

        norm_sq = jnp.sum(jnp.square(M@v))

        return norm_sq - self.eps, norm_sq


class VolumeConstraint(ScalarOptimizationComponent):

    def __call__(self, x, grad):

        (c, cs), dc_dx = jax.value_and_grad(self.eval, has_aux=True)(x)

        if grad.size > 0:
            grad[:] = dc_dx

        id = self.__str__()
        self.ops.update_evals_and_plot(id, cs)

        if not self.silent:
            logger.info(f"{self.__str__()} g(x): {cs:.4f}")

        stop_on_nan(c)
        return float(c)

    def eval(self, x):
        # currently assume all cells have same volume
        V = jnp.mean(self.filter_and_project(x))
        return V - self.eps, V

    def __str__(self):
        return fr"$V \geq {self.eps:2f}$"

        # class EnergyConstraints:

        #     def __init__(self, a, basis_v, extremal_mode, ops, eps=1e-6, verbose=True):
        #         self.a = a
        #         self.basis_v = basis_v
        #         self.extremal_mode = extremal_mode
        #         self.ops = ops
        #         self.verbose = verbose
        #         self.eps = eps

        #         self.n_constraints = 2

        #     def __call__(self, results, x, grad, dummy_run=False):

        #         def obj(C):
        #             m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
        #             C = m @ C @ m
        #             S = jnp.linalg.inv(C)

        #             if self.extremal_mode == 2:
        #                 C, S = S, C

        #             vCv = self.basis_v.T @ C @ self.basis_v
        #             c1, c2, c3 = vCv[0, 0], vCv[1, 1], vCv[2, 2]
        #             return jnp.log10(jnp.array([c1/c2*self.a, c1/c3*self.a]))

        #         Chom, dxfem_dx_vjp, dChom_dxfem = self.ops.Chom, self.ops.dxfem_dx_vjp, self.ops.dChom_dxfem

        #         c = obj(jnp.asarray(Chom))
        #         results[:] = c

        #         if dummy_run:
        #             return

        #         if grad.size > 0:
        #             dc_dChom = jax.jacrev(obj)(jnp.asarray(
        #                 Chom)).reshape((self.n_constraints, 9))
        #             for n in range(self.n_constraints):
        #                 grad[n, :] = dxfem_dx_vjp(dc_dChom[n, :] @ dChom_dxfem)[0]

        #         if self.verbose:
        #             print(f"EnergyConstraints value(s): {c}")

        # class IsotropicConstraint:

        #     def __init__(self, eps, ops, verbose=True):
        #         self.eps = eps
        #         self.ops = ops
        #         self.verbose = verbose

        #     def __call__(self, x, grad):

        #         Chom = self.ops.Chom
        #         dChom_dxfem = self.ops.dChom_dxfem
        #         dxfem_dx_vjp = self.ops.dxfem_dx_vjp

        #         def g(C):
        #             Ciso = self._compute_Ciso(C)
        #             diff = Ciso - C
        #             return jnp.sum(diff**2) / Ciso[0, 0]**2

        #         c, dc_dChom = jax.value_and_grad(g)(jnp.asarray(Chom))

        #         if grad.size > 0:
        #             grad[:] = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]

        #         if self.verbose == True:
        #             print(
        #                 f"- Isotropic Constraint: {c:.2e} (Target ≤{self.eps:}) [{'Satisfied' if c <= self.eps else 'Not Satisfied'}]")

        #         return float(c - self.eps)

        #     def _compute_Ciso(self, C):
        #         Ciso_11 = Ciso_22 = (C[0, 0] + C[1, 1]) / 2.
        #         Ciso_12 = Ciso_21 = C[0, 1]
        #         Ciso_33 = (Ciso_11 - Ciso_12) / 2.

        #         return jnp.array([[Ciso_11, Ciso_12, 0.],
        #                           [Ciso_21, Ciso_22, 0.],
        #                           [0.,      0.,      Ciso_33]])

        # class BulkModulusConstraint:

        #     def __init__(self, base_E, base_nu, a, ops, verbose=True):
        #         self.base_E = base_E
        #         self.base_nu = base_nu
        #         self.base_K = self.compute_K(self.base_E, self.base_nu)
        #         self.a = a
        #         self.aK = self.base_K * self.a
        #         self.ops = ops
        #         self.verbose = verbose
        #         self.n_constraints = 1

        #     def __call__(self, x, grad):

        #         Chom = self.ops.Chom
        #         dChom_dxfem = self.ops.dChom_dxfem
        #         dxfem_dx_vjp = self.ops.dxfem_dx_vjp

        #         def g(C):
        #             S = jnp.linalg.inv(C)
        #             return -1. / (S[0][0] + S[0][1]) / 2.
        #         c, dc_dChom = jax.value_and_grad(g)(jnp.asarray(Chom))

        #         if grad.size > 0:
        #             grad[:] = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]

        #         if self.verbose == True:
        #             print(
        #                 f"- Bulk Modulus: {-c:.2e} (Target ≥{self.aK:.2e}) [{'Satisfied' if -c >= self.aK else 'Not Satisfied'}]")
        # # ≤

        #         return float(self.aK + c)

        #     def compute_K(self, E, nu):
        #         # computes plane stress bulk modulus from E and nu
        #         K = E / (3 * (1 - 2 * nu))
        #         G = E / (2 * (1 + nu))
        #         K_plane = 9.*K*G / (3.*K + 4.*G)
        #         return K_plane

        # class ShearModulusConstraint:

        #     def __init__(self, E_max, nu, ops, a=0.002, verbose=True):
        #         self.E_max = E_max
        #         self.nu = nu
        #         self.G_max = E_max / (2 * (1 + nu))
        #         self.a = a
        #         self.aG = self.G_max * self.a

        #         self.ops = ops
        #         self.verbose = verbose

        #     def __call__(self, x, grad):

        #         Chom = self.ops.Chom
        #         dChom_dxfem = self.ops.dChom_dxfem
        #         dxfem_dx_vjp = self.ops.dxfem_dx_vjp

        #         # g = lambda C: -C[2][2]
        #         def g(C):
        #             S = jnp.linalg.inv(C)
        #             return -1/S[2][2]
        #         c, dc_dChom = jax.value_and_grad(g)(jnp.asarray(Chom))

        #         if grad.size > 0:
        #             grad[:] = dxfem_dx_vjp(np.asarray(
        #                 dc_dChom).flatten() @ dChom_dxfem)[0]

        #         if self.verbose == True:
        #             print(
        #                 f"- Shear Modulus: {-c:.2e} (Target ≥{self.aG:.2e}) [{'Satisfied' if -c >= self.aG else 'Not Satisfied'}]")

        #         return self.aG + float(c)

        # class VolumeConstraint:

        #     def __init__(self, V, ops, verbose=True):
        #         self.V = V
        #         self.evals = []
        #         self.ops = ops
        #         self.verbose = verbose

        #     def __call__(self, x, grad):

        #         # x_fem = self.ops.x_fem
        #         filt_fn = self.ops.filt_fn
        #         beta = self.ops.beta
        #         eta = self.ops.eta

        #         # we only constrain the volume of the projected density field per Wang et al. 2011. Right now x_fem sometimes I have a SIMP applied to it, so we do our our filter and projection here. If we remove the SIMP in the Objective function in the future we could use this commented out code b/c x_fem final step would be just the projection
        #         # x_fem = self.ops.x_fem
        #         # volume, dvdx = jax.value_and_grad(lambda x: jnp.mean(x))(x_fem)

        #         def g(x):
        #             x = filt_fn(x)
        #             x = jax_projection(x, beta, eta)
        #             return jnp.mean(x)

        #         volume, dvdx = jax.value_and_grad(g)(x)

        #         if grad.size > 0:
        #             grad[:] = dvdx

        #         if self.verbose == True:
        #             print(
        #                 f"- Volume: {volume:.3f} (Target ≤{self.V}) [{'Satisfied' if volume <= self.V else 'Not Satisfied'}]")

        #         return float(volume) - self.V
