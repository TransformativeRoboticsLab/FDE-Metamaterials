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
from metatop.Metamaterial import Metamaterial


class OptimizationPlot:

    def __init__(self):
        self.fig: plt.Figure = None
        self.ax1: list[plt.Axes] = None
        self.ax2: plt.Axes = None
        self.eval_lines: dict[str, plt.Line2D] = {}

    def setup(self, **kwargs):
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
                     title='Optimization Progress',
                     **kwargs)

        # Draw the figure once to initialize everything
        self.fig.canvas.draw()

    def update_eval_plot(self, component_id: str, evals: list[dict]):
        for n, eval in enumerate(evals):
            id = f"{component_id}_{n}"
            if id not in self.eval_lines:
                line = self.ax2.plot(
                    'x_data', 'y_data', label=eval.get('label', id), marker='.', data=eval)[0]
                self.eval_lines[id] = line
            else:
                self.eval_lines[id].set_data(eval['x_data'], eval['y_data'])

    def update_images(self, projection_function: fe.Function, fields: dict[str, any]):
        if len(fields) != len(self.ax1):
            raise ValueError(
                f"Number of fields ({len(fields):d}) must match number of axes ({len(self.ax1):d})")
        for ax, (name, field) in zip(self.ax1, fields.items()):
            ax.clear()  # Clear the axis completely
            if field.shape[0] == projection_function.function_space().dim():
                projection_function.vector()[:] = field
                self._plot_density(projection_function, title=f"{name}", ax=ax)
            else:
                ax.imshow(field, cmap='gray')
                ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])

    def draw(self):
        logger.debug(f"{self.__class__.__name__} drawing")
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax2.legend(loc='lower right', bbox_to_anchor=(1.1, 1))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
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

        ax.margins(x=0, y=0)

        # quad meshes aren't supported using the standard plot interface but we can convert them to an image and use imshow
        # the ordering of a quad mesh is row-major and imshow expects row-major so it works out
        cell_type = r_in.function_space().ufl_cell().cellname()
        if cell_type == 'quadrilateral':
            r_vec = r.vector()[:]
            # assume square space
            nely = np.sqrt(r_vec.size).astype(int)
            nelx = nely
            img = ax.imshow(r_vec.reshape((nely, nelx)),
                            cmap='gray', vmin=0, vmax=1)
            ax.set_title(title)
            return img

        # We don't return the fenics plot object as it's not compatible with blitting
        fe.plot(r, cmap=cmap, vmin=vmin, vmax=vmax, title=title)
        ax.set_title(title)
        return None


@dataclass
class OptimizationState:
    # Optimization definition values
    basis_v: np.ndarray = None
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
    x_history: list = field(default_factory=lambda: [])

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
    eval_axis_kwargs: dict = field(default_factory=dict)

    # Misc.
    verbose: bool = True
    silent: bool = False
    print_inverval: int = 5

    def __post_init__(self):
        if self.show_plot:
            self.opt_plot.setup(**self.eval_axis_kwargs)

        if self.extremal_mode and self.extremal_mode not in [1, 2]:
            raise ValueError(
                f"Invalid extremal_mode: {self.extremal_mode}. Must be 1 or 2.")

        if self.basis_v is not None and not np.allclose(self.basis_v.T@self.basis_v, np.eye(3)):
            raise ValueError("basis_v is not orthonormal.")

    def update_evals_and_plot(self, component_id: str, c: float | np.ndarray, draw_now: bool = False, labels: list = []):
        self.update_evals(component_id, c)

        is_first_call = self.obj_n_calls == 1
        is_scheduled_update = self.obj_n_calls % self.plot_interval == 0
        is_last_component = component_id == list(self.evals.keys())[-1]

        should_draw = draw_now or (is_first_call or (
            is_scheduled_update and is_last_component))

        if should_draw:
            self.draw(update_images=True)

    def draw(self, update_images: bool = False):
        for comp_id in self.evals.keys():
            self.update_plot(comp_id, update_images)
            update_images = False
        self.opt_plot.draw()

    def update_plot(self, component_id: str, update_images: bool = False, labels: list = []):
        if update_images:
            fields = self._prepare_fields(self.x)
            r = fe.Function(self.metamaterial.R)
            self.opt_plot.update_images(r, fields)

        eval_data = self.evals.get(component_id, None)
        if not eval_data:
            logger.warning(
                f"Component {component_id} not in OPS evals dict.")
            return

        eval_data = np.asarray(eval_data)

        if len(labels) == 0:
            labels = []
            if eval_data.ndim == 1:
                labels.append(component_id)
            elif eval_data.ndim > 1:
                for n in range(eval_data.shape[1]):
                    labels.append(f"{component_id}_{n}")

        evals = []
        # x_data = range(1, self.obj_n_calls+1)
        len_data = eval_data.size if eval_data.ndim == 1 else eval_data.shape[0]
        start_iter = self.obj_n_calls - len_data + 1
        end_iter = self.obj_n_calls + 1
        x_data = range(start_iter, end_iter)
        if eval_data.ndim == 1:
            evals.append({'x_data': x_data,
                          'y_data': eval_data,
                          'label': labels[0]})
        elif eval_data.ndim > 1:
            for n in range(eval_data.shape[1]):
                evals.append({'x_data': x_data,
                              'y_data': eval_data[:, n],
                              'label': labels[n]})

        self.opt_plot.update_eval_plot(component_id, evals)

    def update_state(self, sols, Chom, dChom_dxfem, dxfem_dx_vjp, x, increment_obj_n_calls=True):
        self.sols = sols
        self.Chom = Chom
        self.dChom_dxfem = dChom_dxfem
        self.dxfem_dx_vjp = dxfem_dx_vjp
        self.x = x

        if increment_obj_n_calls:
            self.obj_n_calls += 1

    def update_evals(self, component_id: str, c: Union[float, np.ndarray]):
        if component_id not in self.evals:
            self.evals[component_id] = []
            logger.debug(f"Adding {component_id} to OPS evals dict")

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

    def print_now(self):
        return self.obj_n_calls % self.print_inverval == 0

    def __repr__(self):
        filt_type = type(self.filt).__name__ if self.filt else None
        return (f"OptimizationState(sols={self.sols}, Chom={self.Chom.tolist()}, "
                f"dChom_dxfem={self.dChom_dxfem.tolist()}, dxfem_dx_vjp={self.dxfem_dx_vjp}, "
                f"x={self.x.tolist()}, beta={self.beta}, eta={self.eta}, pen={self.pen}, "
                f"filt={self.filt}, filt_fn={self.filt_fn}, epoch={self.epoch}, "
                f"epoch_iter_tracker={self.epoch_iter_tracker}, evals={self.evals})")

    def __str__(self):
        filt_type = type(self.filt).__name__ if self.filt else "None"
        return (f"Optimization State:\n"
                f"  basis_v: {self.basis_v}\n"
                f"  extremal_mode: {self.extremal_mode}\n"
                f"  sols: {self.sols}\n"
                f"  Chom: {self.Chom.shape}\n"
                f"  dChom_dxfem: {self.dChom_dxfem.shape}\n"
                f"  dxfem_dx_vjp: {self.dxfem_dx_vjp}\n"
                f"  x: {self.x.shape}\n"
                f"  beta: {self.beta}\n"
                f"  eta: {self.eta}\n"
                f"  pen: {self.pen}\n"
                f"  filt: {filt_type}\n"
                f"  filt_fn: {self.filt_fn}\n"
                f"  epoch: {self.epoch}\n"
                f"  epoch_iter_tracker: {self.epoch_iter_tracker}\n"
                f"  evals: {self.evals}")
