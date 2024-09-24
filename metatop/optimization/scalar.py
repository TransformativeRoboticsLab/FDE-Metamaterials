
import fenics as fe
import jax
import jax.interpreters
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from metatop.filters import jax_projection
from metatop.image import bitmapify


class EnergyObjective:
    def __init__(self, v, extremal_mode, metamaterial, ops, verbose=True, plot_interval=10, plot=True):
        self.v = v
        self.extremal_mode = extremal_mode
        self.metamaterial = metamaterial
        self.ops = ops
        self.verbose = verbose
        self.plot_interval = plot_interval
        self.evals = []
        self.n_constraints = 1

        if plot:
            plt.ion()
            self.fig = plt.figure(figsize=(24, 8))
            grid_spec = gridspec.GridSpec(2, 6, )
            self.ax1 = [plt.subplot(grid_spec[0, 0]), plt.subplot(grid_spec[0, 1]), plt.subplot(
                grid_spec[0, 2]), plt.subplot(grid_spec[0, 3]), plt.subplot(grid_spec[0, 4]), plt.subplot(grid_spec[0, 5])]
            self.ax2 = plt.subplot(grid_spec[1, :])
        else:
            self.fig = None

    def __call__(self, x, grad):

        filt_fn, beta, eta = self.ops.filt_fn, self.ops.beta, self.ops.eta

        def filter_and_project(x):
            x = filt_fn(x)
            x = jax_projection(x, beta, eta)
            return x

        x_fem, dxfem_dx_vjp = jax.vjp(filter_and_project, x)

        self.metamaterial.x.vector()[:] = x_fem
        sols, Chom, _ = self.metamaterial.solve()
        E_max, nu = self.metamaterial.prop.E_max, self.metamaterial.prop.nu
        dChom_dxfem = self.metamaterial.homogenized_C(sols, E_max, nu)[1]

        self.ops.update_state(sols, Chom, dChom_dxfem, dxfem_dx_vjp, x_fem)

        def obj(C):
            m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
            C = m @ C @ m
            S = jnp.linalg.inv(C)

            if self.extremal_mode == 2:
                C, S = S, C

            vCv = self.v.T @ C @ self.v
            c1, c2, c3 = vCv[0, 0], vCv[1, 1], vCv[2, 2]
            return jnp.log10(c1**2/c2/c3), jnp.array([c1, c2, c3])

        (c, cs), dc_dChom = jax.value_and_grad(obj, has_aux=True)(jnp.asarray(Chom))

        self.evals.append(c)

        if grad.size > 0:
            g = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]
            grad[:] = g

        if self.verbose:
            print("-" * 30)
            print(
                f"Epoch {self.ops.epoch:d}, Step {len(self.evals):d}, Beta = {self.ops.beta:.1f}, Eta = {self.ops.eta:.1f}")
            print("-" * 30)
            print(f"Energy: {c:.4f}")
            print(f"Actual Values: {cs}")

        if (len(self.evals) % self.plot_interval == 1) and self.fig is not None:
            x_tilde = filt_fn(x)
            x_bar = jax_projection(x_tilde, beta, eta)
            img_resolution = 200
            img_shape = (self.metamaterial.width, self.metamaterial.height)
            r_img = self.metamaterial.x.copy(deepcopy=True)
            x_img = np.flip(bitmapify(r_img, img_shape,
                            (img_resolution, img_resolution)), axis=0)

            fields = {f'x (V={np.mean(x):.3f})': x,
                      f'x_tilde (V={np.mean(x_tilde):.3f})': x_tilde,
                      f'x_bar beta={beta:d} (V={np.mean(x_bar):.3f})': x_bar,
                      #    'grad': g,
                      f'x_fem (V={np.mean(x_fem):.3f})': x_fem,
                      'x_img': x_img,
                      'Tiling': np.tile(x_img, (3, 3))}
            self.update_plot(fields)

        return float(c)

    def update_plot(self, fields):
        if len(fields) != len(self.ax1):
            raise ValueError("Number of fields must match number of axes")

        r = fe.Function(self.metamaterial.R)
        for ax, (name, field) in zip(self.ax1, fields.items()):
            if field.size == self.metamaterial.R.dim():
                r.vector()[:] = field
                cmap = 'gray' if 'x' in name else 'viridis'
                cb = 'x' not in name
                vmin = 0 if 'x' in name else None
                vmax = 1 if 'x' in name else None
                self.plot_density(
                    r, title=f"{name}", ax=ax, cmap=cmap, colorbar=cb, vmin=vmin, vmax=vmax)
            else:
                ax.imshow(255 - field, cmap='gray')
                ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])

        self.ax2.clear()
        f_arr = np.asarray(self.evals)
        self.ax2.plot(range(1, len(self.evals)+1), f_arr, marker='o')
        self.ax2.grid(True)
        self.ax2.set_xlim(left=0, right=len(self.evals) + 2)

        for iter_val in self.ops.epoch_iter_tracker:
            self.ax2.axvline(x=iter_val, color='black',
                             linestyle='--', alpha=0.5, linewidth=3.)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(1e-3)

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

        ax.margins(x=0,y=0)

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

class EnergyConstraints:
    
    def __init__(self, a, v, extremal_mode, ops, eps=1e-6, verbose=True):
        self.a = a
        self.v = v
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
                
            vCv = self.v.T @ C @ self.v
            c1, c2, c3 = vCv[0, 0], vCv[1, 1], vCv[2, 2]
            return jnp.log10(jnp.array([c1/c2*self.a, c1/c3*self.a]))

        Chom, dxfem_dx_vjp, dChom_dxfem = self.ops.Chom, self.ops.dxfem_dx_vjp, self.ops.dChom_dxfem
        
        c = obj(jnp.asarray(Chom))
        results[:] = c
        
        if dummy_run:
            return
        
        if grad.size > 0:
            dc_dChom = jax.jacrev(obj)(jnp.asarray(Chom)).reshape((self.n_constraints, 9))
            for n in range(self.n_constraints):
                grad[n,:] = dxfem_dx_vjp(dc_dChom[n, :] @ dChom_dxfem)[0]

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

