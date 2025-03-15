import abc

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from nlopt import ForcedStop

from metatop.filters import jax_projection


class OptimizationComponent(abc.ABC):

    # TODO: Change verbose to default False eventually
    def __init__(self, basis_v, extremal_mode, metamaterial, ops, verbose=True, silent=False):
        # Mutables
        self.verbose = verbose
        self.silent = silent

        # Immutables
        self._basis_v = basis_v
        self._extremal_mode = extremal_mode
        self._metamaterial = metamaterial
        self._ops = ops

        if self.verbose:
            logger.info(f"Initializing {self.__class__.__name__}")
            logger.debug(f"V:\n{self.basis_v}")
            logger.debug(f"m: {self.extremal_mode}")
            logger.debug(f"OPS: {self.ops}")

    @abc.abstractmethod
    def eval(self, C):
        pass

    @abc.abstractmethod
    def adjoint(self, dc_dChom, dChom_dxfem, dxfem_dx_vjp):
        pass

    @property
    def basis_v(self):
        return self._basis_v

    @property
    def extremal_mode(self):
        return self._extremal_mode

    @property
    def metamaterial(self):
        return self._metamaterial

    @property
    def ops(self):
        return self._ops

    @property
    @abc.abstractmethod
    def n_constraints(self):
        pass

    def forward(self, x):
        x_fem, dxfem_dx_vjp = jax.vjp(self.filter_and_project, x)

        self.metamaterial.x.vector()[:] = x_fem
        sols, Chom, _ = self.metamaterial.solve()
        Chom = jnp.asarray(Chom)
        E_max, nu = self.metamaterial.prop.E_max, self.metamaterial.prop.nu
        dChom_dxfem = self.metamaterial.homogenized_C(sols, E_max, nu)[1]

        self.ops.update_state(sols, Chom, dChom_dxfem, dxfem_dx_vjp, x_fem)
        return Chom, dChom_dxfem, dxfem_dx_vjp

    def filter_and_project(self, x):
        x = self.ops.filt_fn(x)
        x = jax_projection(x, self.ops.beta, self.ops.eta)
        return x


class ScalarOptimizationComponent(OptimizationComponent):

    @abc.abstractmethod
    def __call__(self, x, grad):
        pass

    @property
    def n_constraints(self):
        return 1

    def update_metrics(self, c, cs=None):
        self.ops.evals[-1].append(c)
        if self.silent:
            return

        logger.info(
            f"E: {self.ops.epoch:d} - S: {len(self.ops.evals):d} - f(x): {c:.4f}")
        # TODO: Implement later
        # if self.verbose:
        #     print("-" * 50)
        #     print(
        #         f"Epoch {self.ops.epoch:d}, Step {len(self.ops.evals):d}, Beta = {self.ops.beta:.1f}, Eta = {self.ops.eta:.1f}")
        #     print("-" * 30)
        #     print(f"f(x): {c:.4f}")
        #     print(f"Rayleigh Quotients: {cs}")
        #     e, v = np.linalg.eigh(mandelize(self.ops.Chom))
        #     print(f"Actual (normed) eigenvalues: {e/np.max(e)}")
        #     print(f"Desired eigenvectors:")
        #     print(self.basis_v)
        #     print(f"Actual eigenvectors:")
        #     print(v)
        #     # print(f"")
        # else:
        #     print(f"{len(self.ops.evals):04d} - f(x) = {c}")


class VectorOptimizationComponent(OptimizationComponent):

    @abc.abstractmethod
    def __call__(self, results, x, grad):
        pass


def stop_on_nan(x):
    if np.isnan(x).any():
        logger.error(
            "NaN value detected in objective function. Terminating optimization run.")
        raise ForcedStop
