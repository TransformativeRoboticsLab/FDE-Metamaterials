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

from .utils import *


class RayleighScalarObjective(ScalarOptimizationComponent):
    def __init__(self, basis_v: np.ndarray, extremal_mode: int, metamaterial: int, ops: OptimizationState, verbose: bool = True, plot_interval: int = 25, show_plot: bool = True, img_resolution: tuple[int, int] = (200, 200), eps: float = 1., silent: bool = False):
        super().__init__(basis_v=basis_v,
                         extremal_mode=extremal_mode,
                         metamaterial=metamaterial,
                         ops=ops,
                         verbose=verbose)

        self.eps = eps

        self.plot_interval = plot_interval
        self.show_plot = show_plot
        self.silent = silent
        self.img_resolution = img_resolution
        self.img_shape = (self.metamaterial.width, self.metamaterial.height)

        self.fig = None
        self.epoch_lines = []
        self.last_epoch_plotted = -1

    def __call__(self, x, grad):

        sols, Chom, dChom_dxfem, dxfem_dx_vjp = self.forward(x)

        (c, cs), dc_dChom = jax.value_and_grad(
            self.eval, has_aux=True)(Chom)

        if grad.size > 0:
            grad[:] = self.adjoint(dc_dChom, dChom_dxfem, dxfem_dx_vjp)
            # name = self.__class__.__name__
            # logger.debug(f"{name} Grad max: {np.max(grad):.4f}")
            # logger.debug(f"{name} Grad min: {np.min(grad):.4f}")
            # logger.debug(f"{name} Grad Norm {np.linalg.norm(grad):.4f}")

        self.ops.update_evals(self.__str__(), float(c))
        self.ops.update_state(sols, Chom, dChom_dxfem, dxfem_dx_vjp, x)
        self.ops.update_plot(component_id=self.__str__(), is_primary=True)

        logger.info(f"{self.ops.obj_n_calls} -- {float(c):.4f}")

        stop_on_nan(c)
        return float(c)

    def eval(self, C):
        M = mandelize(C)
        # M = jnp.linalg.inv(M) if self.extremal_mode == 2 else M
        M /= jnorm(M, ord=2)

        r1, r2, r3 = ray_q(M, self.basis_v)
        amean = (r2 + r3) / 2
        gmean = jnp.sqrt(r2*r3)
        hmean = 2 / (1/r2 + 1/r3)
        return (-1)**(self.extremal_mode-1)*r1/amean, jnp.array([r1, r2, r3])

    def adjoint(self, dc_dChom, dChom_dxfem, dxfem_dx_vjp):
        return dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]

    def __str__(self):
        return self.__class__.__name__


class EigenvectorConstraint(ScalarOptimizationComponent):

    def __init__(self, basis_v, extremal_mode, metamaterial, ops, verbose=False):
        super().__init__(basis_v=basis_v,
                         extremal_mode=extremal_mode,
                         metamaterial=metamaterial,
                         ops=ops,
                         verbose=verbose)

    def __call__(self, _, grad):

        Chom, dxfem_dx_vjp, dChom_dxfem = self.ops.Chom, self.ops.dxfem_dx_vjp, self.ops.dChom_dxfem

        c, dc_dChom = jax.value_and_grad(self.eval)(Chom)

        if grad.size > 0:
            grad[:] = self.adjoint(dxfem_dx_vjp, dChom_dxfem, dc_dChom)
            # print(f"Grad max: {np.max(grad)}")
            # print(f"Grad min: {np.min(grad)}")
            # print(f"Grad norm: {np.linalg.norm(grad)}")

        self.ops.update_evals(component_id=self.__str__(), c=c)
        self.ops.update_plot(component_id=self.__str__())

        stop_on_nan(c)
        return float(c)

    def eval(self, C):
        M = mandelize(C)
        # M = jnp.linalg.norm(M) if self.extremal_mode == 2 else M
        M /= jnorm(M)

        e_vals, e_vecs = jnp.linalg.eigh(M)
        V = self.basis_v
        v1, v2, v3 = V[:, 0], V[:, 1], V[:, 2]
        outer_v1 = jnp.outer(v1, v1)
        outer_v2 = jnp.outer(v2, v2)
        outer_v3 = jnp.outer(v3, v3)

        min_penalty = jnp.inf
        for p in itertools.permutations([0, 1, 2]):
            current_penalty = 0.0
            current_penalty += jnp.linalg.norm(outer_v1 - jnp.outer(
                e_vecs[:, p[0]], e_vecs[:, p[0]]), ord='fro')**2
            current_penalty += jnp.linalg.norm(outer_v2 - jnp.outer(
                e_vecs[:, p[1]], e_vecs[:, p[1]]), ord='fro')**2
            current_penalty += jnp.linalg.norm(outer_v3 - jnp.outer(
                e_vecs[:, p[2]], e_vecs[:, p[2]]), ord='fro')**2
            min_penalty = jnp.minimum(min_penalty, current_penalty)

        return min_penalty

    def adjoint(self, dxfem_dx_vjp, dChom_dxfem, dc_dChom):
        return dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]

    def __str__(self):
        return self.__class__.__name__


class EnergyConstraints:

    def __init__(self, a, basis_v, extremal_mode, ops, eps=1e-6, verbose=True):
        self.a = a
        self.basis_v = basis_v
        self.extremal_mode = extremal_mode
        self.ops = ops
        self.verbose = verbose
        self.eps = eps

        self.n_constraints = 2

    def __call__(self, results, x, grad, dummy_run=False):

        def obj(C):
            m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
            C = m @ C @ m
            S = jnp.linalg.inv(C)

            if self.extremal_mode == 2:
                C, S = S, C

            vCv = self.basis_v.T @ C @ self.basis_v
            c1, c2, c3 = vCv[0, 0], vCv[1, 1], vCv[2, 2]
            return jnp.log10(jnp.array([c1/c2*self.a, c1/c3*self.a]))

        Chom, dxfem_dx_vjp, dChom_dxfem = self.ops.Chom, self.ops.dxfem_dx_vjp, self.ops.dChom_dxfem

        c = obj(jnp.asarray(Chom))
        results[:] = c

        if dummy_run:
            return

        if grad.size > 0:
            dc_dChom = jax.jacrev(obj)(jnp.asarray(
                Chom)).reshape((self.n_constraints, 9))
            for n in range(self.n_constraints):
                grad[n, :] = dxfem_dx_vjp(dc_dChom[n, :] @ dChom_dxfem)[0]

        if self.verbose:
            print(f"EnergyConstraints value(s): {c}")


class IsotropicConstraint:

    def __init__(self, eps, ops, verbose=True):
        self.eps = eps
        self.ops = ops
        self.verbose = verbose

    def __call__(self, x, grad):

        Chom = self.ops.Chom
        dChom_dxfem = self.ops.dChom_dxfem
        dxfem_dx_vjp = self.ops.dxfem_dx_vjp

        def g(C):
            Ciso = self._compute_Ciso(C)
            diff = Ciso - C
            return jnp.sum(diff**2) / Ciso[0, 0]**2

        c, dc_dChom = jax.value_and_grad(g)(jnp.asarray(Chom))

        if grad.size > 0:
            grad[:] = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]

        if self.verbose == True:
            print(
                f"- Isotropic Constraint: {c:.2e} (Target ≤{self.eps:}) [{'Satisfied' if c <= self.eps else 'Not Satisfied'}]")

        return float(c - self.eps)

    def _compute_Ciso(self, C):
        Ciso_11 = Ciso_22 = (C[0, 0] + C[1, 1]) / 2.
        Ciso_12 = Ciso_21 = C[0, 1]
        Ciso_33 = (Ciso_11 - Ciso_12) / 2.

        return jnp.array([[Ciso_11, Ciso_12, 0.],
                          [Ciso_21, Ciso_22, 0.],
                          [0.,      0.,      Ciso_33]])


class BulkModulusConstraint:

    def __init__(self, base_E, base_nu, a, ops, verbose=True):
        self.base_E = base_E
        self.base_nu = base_nu
        self.base_K = self.compute_K(self.base_E, self.base_nu)
        self.a = a
        self.aK = self.base_K * self.a
        self.ops = ops
        self.verbose = verbose
        self.n_constraints = 1

    def __call__(self, x, grad):

        Chom = self.ops.Chom
        dChom_dxfem = self.ops.dChom_dxfem
        dxfem_dx_vjp = self.ops.dxfem_dx_vjp

        def g(C):
            S = jnp.linalg.inv(C)
            return -1. / (S[0][0] + S[0][1]) / 2.
        c, dc_dChom = jax.value_and_grad(g)(jnp.asarray(Chom))

        if grad.size > 0:
            grad[:] = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]

        if self.verbose == True:
            print(
                f"- Bulk Modulus: {-c:.2e} (Target ≥{self.aK:.2e}) [{'Satisfied' if -c >= self.aK else 'Not Satisfied'}]")
# ≤

        return float(self.aK + c)

    def compute_K(self, E, nu):
        # computes plane stress bulk modulus from E and nu
        K = E / (3 * (1 - 2 * nu))
        G = E / (2 * (1 + nu))
        K_plane = 9.*K*G / (3.*K + 4.*G)
        return K_plane


class ShearModulusConstraint:

    def __init__(self, E_max, nu, ops, a=0.002, verbose=True):
        self.E_max = E_max
        self.nu = nu
        self.G_max = E_max / (2 * (1 + nu))
        self.a = a
        self.aG = self.G_max * self.a

        self.ops = ops
        self.verbose = verbose

    def __call__(self, x, grad):

        Chom = self.ops.Chom
        dChom_dxfem = self.ops.dChom_dxfem
        dxfem_dx_vjp = self.ops.dxfem_dx_vjp

        # g = lambda C: -C[2][2]
        def g(C):
            S = jnp.linalg.inv(C)
            return -1/S[2][2]
        c, dc_dChom = jax.value_and_grad(g)(jnp.asarray(Chom))

        if grad.size > 0:
            grad[:] = dxfem_dx_vjp(np.asarray(
                dc_dChom).flatten() @ dChom_dxfem)[0]

        if self.verbose == True:
            print(
                f"- Shear Modulus: {-c:.2e} (Target ≥{self.aG:.2e}) [{'Satisfied' if -c >= self.aG else 'Not Satisfied'}]")

        return self.aG + float(c)


class VolumeConstraint:

    def __init__(self, V, ops, verbose=True):
        self.V = V
        self.evals = []
        self.ops = ops
        self.verbose = verbose

    def __call__(self, x, grad):

        # x_fem = self.ops.x_fem
        filt_fn = self.ops.filt_fn
        beta = self.ops.beta
        eta = self.ops.eta

        # we only constrain the volume of the projected density field per Wang et al. 2011. Right now x_fem sometimes I have a SIMP applied to it, so we do our our filter and projection here. If we remove the SIMP in the Objective function in the future we could use this commented out code b/c x_fem final step would be just the projection
        # x_fem = self.ops.x_fem
        # volume, dvdx = jax.value_and_grad(lambda x: jnp.mean(x))(x_fem)

        def g(x):
            x = filt_fn(x)
            x = jax_projection(x, beta, eta)
            return jnp.mean(x)

        volume, dvdx = jax.value_and_grad(g)(x)

        if grad.size > 0:
            grad[:] = dvdx

        if self.verbose == True:
            print(
                f"- Volume: {volume:.3f} (Target ≤{self.V}) [{'Satisfied' if volume <= self.V else 'Not Satisfied'}]")

        return float(volume) - self.V
