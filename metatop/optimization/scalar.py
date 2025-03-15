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

        sols, Chom, dChom_dxfem, dxfem_dx_vjp, x_fem = self.forward(x)

        (c, cs), dc_dChom = jax.value_and_grad(
            self.eval, has_aux=True)(Chom)

        if grad.size > 0:
            grad[:] = self.adjoint(dc_dChom, dChom_dxfem, dxfem_dx_vjp)
            name = self.__class__.__name__
            logger.debug(f"{name} Grad max: {np.max(grad):.4f}")
            logger.debug(f"{name} Grad min: {np.min(grad):.4f}")
            logger.debug(f"{name} Grad Norm {np.linalg.norm(grad):.4f}")

        self.ops.update_evals(self.__str__(), float(c))
        self.ops.update_state(sols, Chom, dChom_dxfem, dxfem_dx_vjp, x_fem)

        if self.ops.obj_n_calls % self.plot_interval == 1:
            self.update_plot(x)

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

    def update_plot(self, x):
        if self.fig is None:
            self._setup_plots()

        if self.fig is None:
            return

        fields = self._prepare_fields(x)
        self._update_image_plots(fields)
        self._update_evaluation_plot()

        if self.show_plot:
            self.fig.canvas.draw()
            plt.pause(1e-3)

    def _prepare_fields(self, x):
        filt_fn, beta, eta = self.ops.filt_fn, self.ops.beta, self.ops.eta
        x_tilde = filt_fn(x)
        x_bar = jax_projection(x_tilde, beta, eta)
        x_img = bitmapify(self.metamaterial.x.copy(
            deepcopy=True), self.img_shape, self.img_resolution, invert=True)
        fields = {r'$\rho$': x,
                  r'$\tilde{\rho}$': x_tilde,
                  fr'$\bar{{\rho}}$ ($\beta$={int(beta):d})': x_bar,
                  r'$\bar{\rho}$ bitmap': x_img,
                  'Image tiling': np.tile(x_img, (3, 3))}
        if len(fields) != len(self.ax1):
            raise ValueError(
                f"Number of fields ({len(fields):d}) must match number of axes ({len(self.ax1):d})")
        return fields

    def _update_image_plots(self, fields):
        r = fe.Function(self.metamaterial.R)
        for ax, (name, field) in zip(self.ax1, fields.items()):
            if field.shape[0] == self.metamaterial.R.dim():
                r.vector()[:] = field
                self.plot_density(r, title=f"{name}", ax=ax)
            else:
                ax.imshow(field, cmap='gray')
                ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])

    def _update_evaluation_plot(self):
        x_data = range(1, self.ops.obj_n_calls+1)
        y_data = np.asarray(self.ops.evals[self.__str__()])

        self.eval_line.set_data(x_data, y_data)

        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax2.set_xlim(left=0, right=self.ops.obj_n_calls+2)

        for idx in self.ops.epoch_iter_tracker:
            if idx > self.last_epoch_plotted:
                self.last_epoch_plotted = idx
                self.epoch_lines.append(self.ax2.axvline(x=idx,
                                                         color='k',
                                                         linestyle='--',
                                                         alpha=0.5,
                                                         linewidth=3.))

    def _setup_plots(self):
        plt.ion() if self.show_plot else plt.ioff()
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
                     ylabel='f(x)',
                     xlim=(0, 10),
                     title='Optimization Progress')

        self.eval_line = self.ax2.plot([0], [0], label='$f(x)$')[0]

    def plot_density(self, r_in, cmap='gray', vmin=0, vmax=1, title=None, ax=None, colorbar=False):
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
            print(f"Grad max: {np.max(grad)}")
            print(f"Grad min: {np.min(grad)}")
            print(f"Grad norm: {np.linalg.norm(grad)}")

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

    def update_metrics(self, c):
        print(f"g(x): {c:.4f}")


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
