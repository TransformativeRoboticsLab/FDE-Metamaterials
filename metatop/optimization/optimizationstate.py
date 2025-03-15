from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Union

import numpy as np

from metatop.filters import DensityFilter, HelmholtzFilter


@dataclass
class OptimizationState:
    sols: list = field(default_factory=list)
    Chom: np.array = field(default_factory=lambda: np.zeros((3, 3)))
    dChom_dxfem: np.array = field(default_factory=lambda: np.zeros((3, 3, 1)))
    dxfem_dx_vjp: Callable[[np.ndarray], np.ndarray] = None
    xfem: np.array = field(default_factory=lambda: np.zeros(1))
    beta: float = 1.
    eta:  float = 0.5
    pen:  float = 3.  # vestigial from other optimization types that use SIMP
    filt: Union[DensityFilter, HelmholtzFilter] = None
    filt_fn: partial = None
    epoch: int = 0
    epoch_iter_tracker: list = field(default_factory=list)
    evals: list = field(default_factory=lambda: [[]])

    def update_state(self, sols, Chom, dChom_dxfem, dxfem_dx_vjp, x_fem):
        self.sols = sols
        self.Chom = Chom
        self.dChom_dxfem = dChom_dxfem
        self.dxfem_dx_vjp = dxfem_dx_vjp
        self.x_fem = x_fem

    def __repr__(self):
        filt_type = type(self.filt).__name__ if self.filt else None
        return (f"OptimizationState(sols={self.sols}, Chom={self.Chom.tolist()}, "
                f"dChom_dxfem={self.dChom_dxfem.tolist()}, dxfem_dx_vjp={self.dxfem_dx_vjp}, "
                f"xfem={self.xfem.tolist()}, beta={self.beta}, eta={self.eta}, pen={self.pen}, "
                f"filt={self.filt}, filt_fn={self.filt_fn}, epoch={self.epoch}, "
                f"epoch_iter_tracker={self.epoch_iter_tracker}, evals={self.evals})")

    def __str__(self):
        filt_type = type(self.filt).__name__ if self.filt else "None"
        return (f"Optimization State:\n"
                f"  sols: {self.sols}\n"
                f"  Chom: {self.Chom.shape}\n"
                f"  dChom_dxfem: {self.dChom_dxfem.shape}\n"
                f"  dxfem_dx_vjp: {self.dxfem_dx_vjp}\n"
                f"  xfem: {self.xfem.shape}\n"
                f"  beta: {self.beta}\n"
                f"  eta: {self.eta}\n"
                f"  pen: {self.pen}\n"
                f"  filt: {filt_type}\n"
                f"  filt_fn: {self.filt_fn}\n"
                f"  epoch: {self.epoch}\n"
                f"  epoch_iter_tracker: {self.epoch_iter_tracker}\n"
                f"  evals: {self.evals}")
