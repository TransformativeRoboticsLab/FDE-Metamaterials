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
    pen:  float = 3.  # holdover from other optimization types that use SIMP
    filt: Union[DensityFilter, HelmholtzFilter] = None
    filt_fn: partial = None
    epoch: int = 0
    epoch_iter_tracker: list = field(default_factory=list)

    def update_state(self, sols, Chom, dChom_dxfem, dxfem_dx_vjp, x_fem):
        self.sols = sols
        self.Chom = Chom
        self.dChom_dxfem = dChom_dxfem
        self.dxfem_dx_vjp = dxfem_dx_vjp
        self.x_fem = x_fem