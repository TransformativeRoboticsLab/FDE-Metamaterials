import abc

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from nlopt import ForcedStop

from metatop.filters import jax_projection
from metatop.Metamaterial import Metamaterial
from metatop.optimization import OptimizationState


class OptimizationComponent(abc.ABC):

    # TODO: Change verbose to default False eventually
    def __init__(self, ops: OptimizationState, eps: float = 0., silent: bool = None, verbose: bool = None):
        # Immutables
        self._ops = ops

        # Mutables
        self.silent = silent or self.ops.silent
        self.verbose = verbose or self.ops.verbose
        self.eps = eps

        if self.verbose:
            logger.info(f"Initializing {self.__class__.__name__}")
            logger.debug(self.ops)

    @abc.abstractmethod
    def eval(self, C: jnp.ndarray):
        pass

    @abc.abstractmethod
    def adjoint(self, dc_dChom, dChom_dxfem, dxfem_dx_vjp):
        pass

    @property
    def ops(self):
        return self._ops

    @property
    @abc.abstractmethod
    def n_constraints(self):
        pass

    def forward(self, x: np.ndarray):
        """
        forward is expecting just the density design DOFs
        """
        assert x.size == self.ops.metamaterial.R.dim()
        metamate = self.ops.metamaterial

        x_fem, dxfem_dx_vjp = jax.vjp(self.filter_and_project, x)

        metamate.x.vector()[:] = x_fem
        sols, Chom = metamate.solve()
        Chom = jnp.asarray(Chom)
        dChom_dxfem = metamate.get_dChom(sols)

        return sols, Chom, dChom_dxfem, dxfem_dx_vjp

    def filter_and_project(self, x: np.ndarray):
        x = self.ops.filt_fn(x)
        x = jax_projection(x, self.ops.beta, self.ops.eta)
        return x

    def __str__(self):
        return self.__class__.__name__


class ScalarOptimizationComponent(OptimizationComponent):

    @abc.abstractmethod
    def __call__(self, x: np.ndarray, grad: np.ndarray):
        pass

    @property
    def n_constraints(self):
        return 1


class VectorOptimizationComponent(OptimizationComponent):

    @abc.abstractmethod
    def __call__(self, results: np.ndarray, x: np.ndarray, grad: np.ndarray):
        pass


def stop_on_nan(x: float | np.ndarray):
    if np.isnan(x).any():
        logger.error(
            "NaN value detected in objective function. Terminating optimization run.")
        raise ForcedStop
