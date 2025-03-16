from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Union

import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib import gridspec

from metatop.filters import DensityFilter, HelmholtzFilter, jax_projection
from metatop.image import bitmapify
from metatop.metamaterial import Metamaterial


class OptimizationPlot:

    def __init__(self):

        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.eval_lines: dict[str, plt.Line2D] = {}

    def setup(self):
        if self.fig:
            logger.debug(
                "Optimization plot already created. Why are you calling setup again?")
            return
        plt.ion()
        self.fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(2, 5)
        self.ax1 = [plt.subplot(gs[0, 0]),
                    plt.subplot(gs[0, 1]),
                    plt.subplot(gs[0, 2]),
                    plt.subplot(gs[0, 3]),
                    plt.subplot(gs[0, 4]),
                    ]

        self.ax2 = plt.subplot(gs[1, :])
        self.ax2.grid(True)
        self.ax2.set(xlabel='Iterations',
                     ylabel='Evals',
                     title='Optimization Progress')

    def update_eval_plot(self, component_id: str, eval_data: np.ndarray):
        x_data = range(1, len(eval_data)+1)
        if component_id not in self.eval_lines:
            line = self.ax2.plot(eval_data, label=component_id, marker='.')[0]
            self.eval_lines[component_id] = line
        else:
            self.eval_lines[component_id].set_data(x_data, eval_data)

    def update_images(self, projection_function: fe.Function, fields):
        if len(fields) != len(self.ax1):
            raise ValueError(
                f"Number of fields ({len(fields):d}) must match number of axes ({len(self.ax1):d})")
        for ax, (name, field) in zip(self.ax1, fields.items()):
            if field.shape[0] == projection_function.function_space().dim():
                projection_function.vector()[:] = field
                self._plot_density(projection_function, title=f"{name}", ax=ax)
            else:
                ax.imshow(field, cmap='gray')
                ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])

    def draw(self):
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.fig.canvas.draw()
        plt.pause(1e-3)

    def _plot_density(self, r_in, cmap='gray', vmin=0, vmax=1, title=None, ax=None, colorbar=False):
        r = fe.Function(r_in.function_space())
        r.vector()[:] = r_in.vector()[:]
        if cmap == 'gray':
            r.vector()[:] = 1. - r.vector()[:]
        r.set_allow_extrapolation(True)

        if isinstance(ax, plt.Axes):
            plt.sca(ax)
        else:
            fig, ax = plt.subplots()
        ax.clear()

        ax.margins(x=0, y=0)

        # quad meshes aren't supported using the standard plot interface but we can convert them to an image and use imshow
        # the ordering of a quad mesh is row-major and imshow expects row-major so it works out
        cell_type = r_in.function_space().ufl_cell().cellname()
        if cell_type == 'quadrilateral':
            r_vec = r.vector()[:]
            # assume square space
            nely = np.sqrt(r_vec.size).astype(int)
            nelx = nely
            plt.imshow(r_vec.reshape((nely, nelx)),
                       cmap='gray', vmin=0, vmax=1)
            ax.set_title(title)
            return

        p = fe.plot(r, cmap=cmap, vmin=vmin, vmax=vmax, title=title)


@dataclass
class OptimizationState:
    # Optimization definition values
    basis_v: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    extremal_mode: int = None
    metamaterial: Metamaterial = None
    filt: Union[DensityFilter, HelmholtzFilter] = None
    filt_fn: partial = None

    # Solved values
    sols: list = field(default_factory=list)
    Chom: np.array = field(default_factory=lambda: np.zeros((3, 3)))
    dChom_dxfem: np.array = field(default_factory=lambda: np.zeros((3, 3, 1)))
    dxfem_dx_vjp: Callable[[np.ndarray], np.ndarray] = None
    x: np.array = field(default_factory=lambda: np.zeros(1))

    # Optimization values
    beta: float = 1.
    eta:  float = 0.5
    pen:  float = 3.  # vestigial from other optimization types that use SIMP
    epoch: int = 0
    epoch_iter_tracker: list = field(default_factory=lambda: [1])
    evals: dict = field(default_factory=dict)
    obj_n_calls: int = 0

    # Plotting values
    opt_plot: OptimizationPlot = field(
        default_factory=lambda: OptimizationPlot())
    plot_interval: int = 25
    show_plot: bool = True
    img_shape: tuple[float, float] = (1., 1.)
    img_resolution: tuple[int, int] = (200, 200)

    # Misc.
    verbose: bool = True

    def __post_init__(self):
        if self.show_plot:
            self.opt_plot.setup()

    def update_plot(self, component_id: str, is_primary: bool = False, draw_now=False):
        if not self.show_plot:
            return

        if not (self.show_plot and self.obj_n_calls % self.plot_interval == 1) and not draw_now:
            return

        eval_data = self.evals.get(component_id, None)
        if eval_data is None:
            logger.warning(
                f"Component {component_id} not in OPS evals dict.")
            return

        if is_primary:
            fields = self._prepare_fields(self.x)
            r = fe.Function(self.metamaterial.R)
            self.opt_plot.update_images(r, fields)

        self.opt_plot.update_eval_plot(component_id, eval_data)

        self.opt_plot.draw()

    def update_state(self, sols, Chom, dChom_dxfem, dxfem_dx_vjp, x):
        self.sols = sols
        self.Chom = Chom
        self.dChom_dxfem = dChom_dxfem
        self.dxfem_dx_vjp = dxfem_dx_vjp
        self.x = x
        self.obj_n_calls += 1

    def update_evals(self, component_id: str, c: Union[float, np.ndarray]):
        if component_id not in self.evals:
            self.evals[component_id] = []

        self.evals[component_id].append(c)

    def _prepare_fields(self, x):
        filt_fn, beta, eta = self.filt_fn, self.beta, self.eta
        x_tilde = filt_fn(x)
        x_bar = jax_projection(x_tilde, beta, eta)
        x_img = bitmapify(self.metamaterial.x.copy(deepcopy=True), self.img_shape,
                          self.img_resolution, invert=True)
        fields = {r'$\rho$': x,
                  r'$\tilde{\rho}$': x_tilde,
                  fr'$\bar{{\rho}}$ ($\beta$={int(beta):d})': x_bar,
                  r'$\bar{\rho}$ bitmap': x_img,
                  'Image tiling': np.tile(x_img, (3, 3))}
        return fields

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
