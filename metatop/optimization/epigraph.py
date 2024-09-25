import inspect

import fenics as fe
import jax
import jax.interpreters
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nlopt
import numpy as np
from jax.numpy.linalg import norm as jnorm
from matplotlib import gridspec

from metatop.filters import jax_projection
from metatop.image import bitmapify


class EpigraphOptimizer(nlopt.opt):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.active_constraints = []

    def setup(self):
        print("Setting up optimizer...")
        self.set_min_objective(Epigraph())
        print("Adding constraints...")
        if self.n_constraints > 0:
            for n, g in enumerate(self.active_constraints):
                print(f"Constraint {n+1:d}/{self.n_constraints:d}: {g}")
                args_count = len(inspect.signature(g).parameters)
                if args_count > 2:
                    super().add_inequality_mconstraint(g, np.zeros(g.n_constraints))
                elif args_count == 2:
                    super().add_inequality_constraint(g, 0.)
                else:
                    raise ValueError(f"Constraint function {g} seems to take an incorrect number of arguments")
        else:
            raise Warning("No active constraints set")
        
        self.set_lower_bounds(np.append(np.zeros(self.size - 1), -np.inf))
        self.set_upper_bounds(np.append(np.ones(self.size - 1), np.inf))
        self.set_param('dual_ftol_rel', 1e-6)

    def optimize(self, x):
        self._update_t(x)
        return super().optimize(x)
        
    def _update_t(self, x):
        print(f"Updating t...\nOld t value {x[-1]:.3e}")
        new_t = -np.inf
        x[-1] = 0.
        for g in self.active_constraints:
            if len(inspect.signature(g).parameters) > 2:
                print(f"Accounting for constraint {g} in t update")
                results = np.zeros(g.n_constraints)
                g(results, x, np.array([]), dummy_run=True)
                new_t = max(new_t, *(results))
        x[-1] = new_t
        print(f"New t value: {x[-1]:.3e}")
    
    def add_inequality_mconstraint(self, *args, uses_t=True):
        if uses_t:
            self.active_constraints.append(args[0])
        return super().add_inequality_mconstraint(*args)
    
    def add_inequality_constraint(self, *args, uses_t=True):
        if uses_t:
            self.active_constraints.append(args[0])
        return super().add_inequality_constraint(*args)
    
    def add_equality_constraint(self, *args):
        raise NotImplementedError("Equality constraints are not supported for this optimizer")
    
    def add_equality_mconstraint(self, *args):
        raise NotImplementedError("Equality constraints are not supported for this optimizer")
        
    @property
    def size(self):
        return self.get_dimension()

    @property
    def n_constraints(self):
        return len(self.active_constraints)


class Epigraph:
    """
    A class representing a minimax optimization problem.
    We reformulate the problem as a nonlinear optimization problem by adding a slack variable t as the objective function.

    min_x max{f1, f2, f3}
    s.t. g(x) <= 0

    becomes

    min_{x, t} t
    s.t. f1 <= t
         f2 <= t
         f3 <= t
         g(x) <= 0

    The objective function is simply the slack variable t.
    The gradiant of the objective function is then all zeros with a 1 at the end of the vector.
    """

    def __call__(self, x, grad):
        """
        Evaluates the objective function of the minimax problem.

        Parameters:
        - x: The input vector.
        - grad: The gradient vector.

        Returns:
        - The objective function value.
        """
        t = x[-1]

        if grad.size > 0:
            grad[:-1], grad[-1] = 0., 1.

        return t


class ExtremalConstraints:
    """
    A class representing the original objective functions of the minimax problem.
    """

    def __init__(self, v, extremal_mode, metamaterial, ops, objective_type, w=jnp.ones(3), verbose=True, plot_interval=10, show_plot=True):
        self.v = v
        self.extremal_mode = extremal_mode
        self.metamaterial = metamaterial
        self.ops = ops
        self.objective_type = objective_type
        self.verbose = verbose
        self.plot_interval = plot_interval
        self.w = w
        self.evals = []

        self.n_constraints = 2 if 'ratio' in self.objective_type else 3
        self.eps = 1.

        if show_plot:
            plt.ion()
        self.fig = plt.figure(figsize=(15, 6))
        grid_spec = gridspec.GridSpec(2, 5, )
        self.ax1 = [plt.subplot(grid_spec[0, 0]), 
                    plt.subplot(grid_spec[0, 1]), 
                    plt.subplot(grid_spec[0, 2]), 
                    plt.subplot(grid_spec[0, 3]), 
                    plt.subplot(grid_spec[0, 4]), 
                    ]
        self.ax2 = plt.subplot(grid_spec[1, :])

        if self.verbose:
            print(f"""
MinimaxConstraint initialized with:
v:
{v}
extremal_mode: {self.extremal_mode}
starting beta: {self.ops.beta}
verbose: {self.verbose}
plot_delay: {self.plot_interval}
objective_type: {self.objective_type}
""")

    def __call__(self, results, x, grad, dummy_run=False):

        x, t = x[:-1], x[-1]

        filt_fn, beta, eta = self.ops.filt_fn, self.ops.beta, self.ops.eta

        def filter_and_project(x):
            x = filt_fn(x)
            x = jax_projection(x, beta, eta)
            # x = jax_simp(x, 3.)
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
            # NOTE: normalize the matrix to ensure the largest eigenvalue is 1, exploiting the fact that basis vectors v are unit length. This normalization simplifies controlling the magnitude of eigenvalues, where we aim to align v as the eigenvectors and adjust their associated eigenvalues. We achieve this by normalizing the matrix to its spectral norm. Subsequently, we calculate the norm of each vector in the basis, where the maximum possible value of the norm for C*v_n after normalization is 1. By evaluating 1 - norm(C*v_n), we attempt to maximize the vector's norm, effectively aligning the eigenvectors with the basis vectors while managing the eigenvalue magnitudes. This isn't perfect and v won't necessarily be the eigenvectors of C at the end of the optimization, but we can introduce additional constraints to help out with that if we need to.
            # NOTE: Because of the normalizaton step, we lose track of any kind of real stiffness of the true material. Thus we need to introduce some other kind of constraint on the system to ensure that the unnormalized homogenized C does not approach a trivial solution. One simple constraint to help out is tr(C) >= value. This will ensure that the homogenized C is not too small.
            S = jnp.linalg.inv(C)

            if self.extremal_mode == 2:
                C, S = S, C

            if self.objective_type != 'ratio':
                C /= jnorm(C, ord=2)
                S /= jnorm(S, ord=2)

            v1, v2, v3 = self.v[:, 0], self.v[:, 1], self.v[:, 2]
            Cv1, Cv2, Cv3 = C@v1, C@v2, C@v3
            c1, c2, c3 = v1.T@Cv1, v2.T@Cv2, v3.T@Cv3
            if self.objective_type == 'ray':
                return jnp.log(self.w*jnp.array([c1, 1-c2, 1-c3])), jnp.array([c1, c2, c3])
            elif self.objective_type == 'ray_sq':
                return (jnp.log(self.w*jnp.array([c1**2, (1. - c2**2), (1. - c3**2), ])+1e-8), jnp.array([c1, c2, c3, ]))
            elif self.objective_type == 'norm':
                return jnp.log(self.w*(jnp.array([jnorm(Cv1), 1-jnorm(Cv2), 1-jnorm(Cv3)])+1e-8)), jnp.array([jnorm(Cv1), jnorm(Cv2), jnorm(Cv3)])
            elif self.objective_type == 'norm_sq':
                return (jnp.log(self.w*jnp.array([Cv1@Cv1, 1 - Cv2@Cv2, 1-Cv3@Cv3, ])+1e-8), jnp.array([Cv1@Cv1, Cv2@Cv2, Cv3@Cv3]))
            elif self.objective_type == 'ratio':
                return jnp.log(jnp.array([c1/c2, c1/c3, ])), jnp.array([c1, c2, c3])
            elif self.objective_type == 'ratio_sq':
                return jnp.log(jnp.array([(c1/c2)**2, (c1/c3)**2, ])), jnp.array([c1, c2, c3])
            elif self.objective_type == 'ratio_c1sq':
                return jnp.log(jnp.array([c1**2/c2, c1**2/c3, ])), jnp.array([c1, c2, c3])
            else:
                raise ValueError('Objective type must be either "rayleigh" or "norm"')

        c, cs = obj(jnp.asarray(Chom))
        results[:] = c - t

        if dummy_run:
            return

        if grad.size > 0:
            dc_dChom = jax.jacrev(obj, has_aux=True)(jnp.asarray(Chom))[0].reshape((self.n_constraints, 9))
            for n in range(self.n_constraints):
                grad[n, :-1] = dxfem_dx_vjp(dc_dChom[n, :] @ dChom_dxfem)[0]
                grad[n, -1] = -1.

        self.evals.append([t, *c])
        if self.verbose:
            print("-" * 30)
            print(
                f"Epoch {self.ops.epoch:d}, Step {len(self.evals):d}, Beta = {self.ops.beta:.1f}, Eta = {self.ops.eta:.1f}")
            print("-" * 30)
            # print(f"g(x) = {c:.4f}")
            # print(t, c)
            print(f"t: {t:.3e} g(x): {c}")
            print(f"Actual Values: {cs}")

        if (len(self.evals) % self.plot_interval == 1) and self.fig is not None:
            self.update_plot(x)

    def update_plot(self, x):
        filt_fn, beta, eta = self.ops.filt_fn, self.ops.beta, self.ops.eta
        x_tilde = filt_fn(x)
        x_bar = jax_projection(x_tilde, beta, eta)
        img_resolution = (200, 200)
        img_shape = (self.metamaterial.width, self.metamaterial.height)
        r_img = self.metamaterial.x.copy(deepcopy=True)
        x_img = bitmapify(r_img, img_shape, img_resolution)
        fields = {f'x (V={np.mean(x):.3f})': x,
                    f'x_tilde (V={np.mean(x_tilde):.3f})': x_tilde,
                    f'x_bar beta={int(beta):d} (V={np.mean(x_bar):.3f})': x_bar,
                    f'x_img': x_img,
                    f'Tiling': np.tile(x_img, (3, 3))}
        if len(fields) != len(self.ax1):
            raise ValueError("Number of fields must match number of axes")

        r = fe.Function(self.metamaterial.R)
        for ax, (name, field) in zip(self.ax1, fields.items()):
            if field.shape[0] == self.metamaterial.R.dim():
                r.vector()[:] = field
                self.plot_density(r, title=f"{name}", ax=ax)
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

    def plot_density(self, r_in, title=None, ax=None):
        r = fe.Function(r_in.function_space())
        r.vector()[:] = 1. - r_in.vector()[:]
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
            plt.imshow(r_vec.reshape((nely, nelx)),
                       cmap='gray', vmin=0, vmax=1)
            ax.set_title(title)
            return

        fe.plot(r, cmap='gray', vmin=0, vmax=1, title=title)

    def __str__(self):
        return "ExtremalConstraints"

class SpectralNormConstraint:
    
    def __init__(self, ops, bound=1., verbose=True):
        self.ops = ops
        self.bound = bound
        self.verbose = verbose
    
    def __call__(self, x, grad, ):
        
        Chom, dChom_dxfem, dxfem_dx_vjp  = self.ops.Chom, self.ops.dChom_dxfem, self.ops.dxfem_dx_vjp

        m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
        def g(C):
            C = m @ C @ m
            return jnp.linalg.norm(C, ord=2)
            # return jnp.trace(C)

        c, dc_dChom = jax.value_and_grad(g)(jnp.asarray(Chom))
        
        if grad.size > 0:
            grad[:-1] = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]
            grad[-1] = 0.

        if self.verbose:
            print(f"Spectral Norm Constraint:")
            print(f"Value: {c:.3f} (Target >={self.bound:.3f})")
            # print(f"Eigenvalues: {np.linalg.eigvalsh(m@Chom@m)}")
        return float(self.bound - c)

    def __str__(self):
        return "SpectralNormConstraint"

class VectorConstraint:
    '''
    This is a generalized class that will handle vector constraints for nlopt.
    It can also be used for a scalar constraint with vector of length 1.

    A constraint is formulated as g_k(x) <= b_k(x).
    We rearrange to give nlopt the form g_k(x) - b_k(x) <= 0.
    The simplified case is where b_k(x) = 0.
    This is the assumed default b/c we set dbdt = 0. on initialization.

    One thing we may wish to do is in a minimax problem we can bound using the optimization variable `t`.
    The interface for constraints and bounds takes this into account, but it could be more general in the future.
    e.g. right now the bounds functions are assumed to only take t, but something like b_k(Chom, x, t) and discarding unused inputs is viable.

    We use a global Chom, r and vjp here because it prevents us from having to rerun the forward pass again for the constraint.
    If we wrapped everything into one large class this could just be a class attribute that gets passed around.
    '''

    def __init__(self, ops, eps=1e-6, verbose=False):
        self.ops = ops
        self.constraints = []
        self.eps = eps
        self.verbose = verbose
        self.dbdt = np.zeros(self.n_constraints)
        self.bounds = None

    def __call__(self, results, x, grad, dummy_run=False):

        x, t = x[:-1], x[-1]

        m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
        Chom = m @ jnp.asarray(self.ops.Chom) @ m
        dChom_dxfem = self.ops.dChom_dxfem
        dxfem_dx_vjp = self.ops.dxfem_dx_vjp

        # values
        gs = np.array([g(Chom) for g in self.constraints])
        bs = np.array([b(t) for b in self.bounds])
        dg_dChoms = [jax.jacrev(g)(Chom) for g in self.constraints]

        results[:] = gs - bs

        if dummy_run:
            return

        if self.verbose:
            print(f"{self.__str__()} value(s): {gs}")
            print(f"{self.__str__()} bound(s): {bs}")

        if grad.size > 0:
            for n in range(self.n_constraints):
                grad[n, :-1] = dxfem_dx_vjp(dg_dChoms[n].flatten() @ dChom_dxfem)[0]
                grad[n, -1] = -self.dbdt[n]

    @property
    def n_constraints(self):
        return len(self.constraints)

    def __str__(self):
        return "MinimaxConstraints"

class EigenvectorConstraint:

    def __init__(self, v, ops, eps=1e-3, verbose=True):
        self.v = v
        self.ops = ops
        self.eps = eps
        self.verbose = verbose

        self.n_constraints = 3

    def __call__(self, results, x, grad, dummy_run=False):

        Chom, dChom_dxfem, dxfem_dx_vjp = self.ops.Chom, self.ops.dChom_dxfem, self.ops.dxfem_dx_vjp

        t = x[-1]

        def obj(C):
            m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
            C = m @ C @ m

            C /= jnorm(C, ord=2)

            v1, v2, v3 = self.v[:, 0], self.v[:, 1], self.v[:, 2]
            # Rayleigh quotients
            r1, r2, r3 = v1.T @ C @ v1, v2.T @ C @ v2, v3.T @ C @ v3
            Cv1, Cv2, Cv3 = C @ v1, C @ v2, C @ v3
            x1, x2, x3 = Cv1 - r1*v1, Cv2 - r2*v2, Cv3 - r3*v3

            return jnp.log(jnp.array([x1.T @ x1, x2.T @ x2, x3.T @ x3])/self.eps), jnp.array([x1.T @ x1, x2.T @ x2, x3.T @ x3])

        c, cs = obj(np.asarray(Chom))
        results[:] = c - t

        if dummy_run:
            return

        if grad.size > 0:
            dc_dChom = jax.jacrev(obj, has_aux=True)(jnp.asarray(Chom))[0].reshape((self.n_constraints, 9))
            for n in range(self.n_constraints):
                grad[n, :-1] = dxfem_dx_vjp(dc_dChom[n, :] @ dChom_dxfem)[0]
                grad[n, -1] = -1.
                # grad[n,:] = dxfem_dx_vjp(dc_dChom[n,:] @ dChom_dxfem)[0]

        if self.verbose:
            print(f"Eigenvector Constraint:")
            print(f"Values: {c}")
            print(f"Residuals: {cs}")

    def __str__(self):
        return "EigenvectorConstraint"

class TraceConstraint:

    def __init__(self, ops, bound=3e-1, verbose=True):
        self.ops = ops
        self.verbose = verbose
        self.bound = bound

    def __call__(self, x, grad):

        Chom, dChom_dxfem, dxfem_dx_vjp = self.ops.Chom, self.ops.dChom_dxfem, self.ops.dxfem_dx_vjp

        m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
        def obj(C):
            return -jnp.trace(m@C@m)

        c, dc_dChom = jax.value_and_grad(obj)(Chom)

        if grad.size > 0:
            grad[:-1] = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]
            grad[-1]  = 0.

        if self.verbose:
            print(f"Trace: {-c:.3f} (Target >={self.bound:.3f})")
            # w,v = np.linalg.eigh(m@Chom@m)
            # print(f"Eigenvalues: {w}")
            # print(f"Rel. Eigenvalues: {w/np.max(w)}")
            # print(f"Sum(w): {np.sum(w):.3f}")
            # print(f"Eigenvectors:\n{v}")

        return float(self.bound + c)

    def __str__(self):
        return "TraceConstraint"

class InvariantsConstraint:

    def __init__(self, ops, verbose=True):
        self.ops = ops
        self.verbose = verbose
        # Invariant bounds:
        # tr(C) >= eps --> eps - tr(C) <= 0 --> (-tr(C)) - (-eps) <= 0
        self.eps = np.array([-3e-1, -0., 1e-1])

        self.n_constraints = 3

        assert self.eps.size == self.n_constraints, "Epsilons must be the same length as the number of constraints"

    def __call__(self, results, x, grad, dummy_run=False):

        Chom, dChom_dxfem, dxfem_dx_vjp = self.ops.Chom, self.ops.dChom_dxfem, self.ops.dxfem_dx_vjp

        t = x[-1]

        def obj(C):
            m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
            C = m @ C @ m
            I1 = jnp.trace(C)
            I2 = 0.5 * (jnp.trace(C)**2 - jnp.trace(C @ C))
            I3 = jnp.linalg.det(C)
            return jnp.array([-I1, -I2, I3])

        c = obj(jnp.asarray(Chom))
        results[:] = c - self.eps

        if dummy_run:
            return

        if grad.size > 0:
            dc_dChom = jax.jacrev(obj)(jnp.asarray(Chom)).reshape((self.n_constraints, 9))
            for n in range(self.n_constraints):
                grad[n, :-1] = dxfem_dx_vjp(dc_dChom[n, :] @ dChom_dxfem)[0]
                grad[n, -1] = 0.

        if self.verbose:
            print(f"Invariant Constraint:")
            print(f"Trace: {-c[0]:.3f} (Target >={-self.eps[0]:.3f})")
            print(
                f"Second Invariant: {-c[1]:.2e} (Target >={-self.eps[1]:.3f})")
            print(f"Det: {c[2]:.2e} (Target <={self.eps[2]:.3f})")


class GeometricConstraints:

    def __init__(self, ops, metamaterial, line_width, line_space, c, eps=1e-3, verbose=True):
        self.ops = ops
        self.metamaterial = metamaterial
        # if 'quad' not in metamaterial.mesh.ufl_cell().cellname():
        # raise ValueError("Geometric Constraints only work with quadrilateral elements")
        self.lw = line_width
        self.ls = line_space
        self.filt_radius = self.ops.filt.radius
        self.eps = eps
        self.verbose = verbose
        self.c = c
        self.n_constraints = 2

        self._eta_e, self._eta_d = self._calculate_etas(
            self.lw, self.ls, self.filt_radius)

        # items to help calculate the gradient of rho_tilde
        self._r_tilde = fe.Function(self.metamaterial.R)

    def __call__(self, results, x, grad, dummy_run=False):

        filt_fn = self.ops.filt_fn
        x, t = x[:-1], x[-1]
        
        def g(x):
            x_tilde = filt_fn(x)
            a1 = jnp.minimum(x_tilde - self._eta_e, 0.)**2
            b1 = self._indicator_fn(x, 'width')
            f, a = plt.subplots(1, 3)
            plt.sca(a[0])
            plt.imshow(a1.reshape((100, 100)))
            plt.colorbar()
            plt.sca(a[1])
            plt.imshow(b1.reshape((100, 100)))
            plt.colorbar()
            plt.sca(a[2])
            plt.imshow((a1*b1).reshape((100, 100)))
            plt.colorbar()
            c1 = jnp.mean(a1*b1)

            a2 = jnp.minimum(self._eta_d - x_tilde, 0.)**2
            b2 = self._indicator_fn(x, 'space')
            c2 = jnp.mean(a2*b2)

            return jnp.log(jnp.array([c1, c2]))

        c = g(x)
        dc_dx = jax.jacrev(g)(x)

        results[:] = c - t*self.eps

        if grad.size > 0:
            for n in range(self.n_constraints):
                grad[n, :-1] = dc_dx[n, :]
                grad[n, -1] = -self.eps

        if self.verbose:
            print(f"Geometric Constraint:")
            print(f"Width: {c[0]:.3e} (Target ≤{t*self.eps:.1e})")
            print(f"Space: {c[1]:.3e} (Target ≤{t*self.eps:.1e})")

    def _indicator_fn(self, x, type):
        if type not in ['width', 'space']:
            raise ValueError("Indicator Function must be 'width' or 'space'")
        filt_fn, beta, eta = self.ops.filt_fn, self.ops.beta, self.ops.eta
        nelx, nely = self.metamaterial.nelx, self.metamaterial.nely

        x_tilde = filt_fn(x)
        x_bar = jax_projection(x_tilde, beta, eta)

        r = fe.Function(self.metamaterial.R)
        r.vector()[:] = x
        # here we use fenics gradient to calculate grad(rho_tilde)
        # first we convert x_tilde the vector to a fenics function in DG space
        self._r_tilde.vector()[:] = x_tilde
        # then we project r_tilde to a CG space so that we can get the gradient
        # this is required because the gradient of a DG function is zero (values are constant across the cell)
        r_tilde_cg = fe.project(self._r_tilde, self.metamaterial.R_cg)
        # now we can calculate the inner product of the gradient of r_tilde and project back to the original DG space
        grad_r_tilde = fe.grad(r_tilde_cg)
        laplac_r_tilde = fe.div(grad_r_tilde)
        grad_r_tilde_norm_sq = fe.project(
            fe.inner(grad_r_tilde, grad_r_tilde), self.metamaterial.R).vector()[:].reshape((nely, nelx))

        r_tilde_img = x_tilde.reshape((nely, nelx))
        def fd_norm_sq(img):
            grad = self._fd_grad(img, h=1 / nelx)
            return grad[0]**2 + grad[1]**2
        fd_grad_r_tilde_norm_sq = fd_norm_sq(r_tilde_img)
        J_fd = jax.jacfwd(fd_norm_sq)(r_tilde_img)
        J_fenics = -2*fe.div(grad_r_tilde)
        # grad_rho_img = self._fd_grad(r_tilde_img, h=1 / nelx)
        # fd_grad_r_tilde_norm_sq = (grad_rho_img[1]**2 + grad_rho_img[0]**2).reshape((nely, nelx))
        # d_check = jax.jacrev(self._fd_grad)(r_tilde_img)
        # d_check = jax.gradient(self._fd_grad)(r_tilde_img)
        # d_check = jax.jacrev(jnp.gradient)(r_tilde_img)
        

        # fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        # plt.sca(ax0)
        # plt.imshow(grad_r_tilde_norm_sq.vector()[
        #            :].reshape((nely, nelx)), cmap='gray')
        # plt.colorbar()
        # plt.title("grad_r_tilde_norm_sq")

        # plt.sca(ax1)
        # plt.imshow(fd_grad_r_tilde_norm_sq, cmap='gray')
        # plt.colorbar()
        # plt.title("fd_grad_r_tilde_norm_sq")

        # plt.sca(ax2)
        # # plt.imshow((grad_x_tilde_norm_sq - grad_r_tilde_norm_sq.vector()[:].reshape((nely, nelx))))
        # diff = fd_grad_r_tilde_norm_sq.flatten(
        # ) - grad_r_tilde_norm_sq.vector()[:]
        # err = fe.Function(self.metamaterial.R)
        # err.vector()[:] = diff
        # plt.plot(diff)
        # # plt.yscale('log')
        # # plt.colorbar()
        # plt.title("diff")
        # plt.show()
        # print(f"Max diff: {np.max(diff)}")
        # print(f"Min diff: {np.min(diff)}")
        # print(f"Mean diff: {np.mean(diff)}")
        # print(f"Std diff: {np.std(diff)}")
        # print(f"Norm diff: {np.linalg.norm(diff)/nelx}")
        # print(f"Fenics norm diff: {norm(err, 'L2')}")

        q = jnp.exp(-self.c * (grad_r_tilde_norm_sq))
        fig, (ax0, ax1) = plt.subplots(1, 2)
        plt.sca(ax0)
        plt.imshow(grad_r_tilde_norm_sq, cmap='gray')
        plt.show()

        if type == 'width':
            return x_bar * q.flatten()
        elif type == 'space':
            return (1. - x_bar) * q.flatten()
        else:
            raise ValueError("Indicator Function must be 'width' or 'space'")

    def _calculate_etas(self, lw, ls, R):
        eta_e, eta_d = 1., 0.
        lwR, lsR = lw/R, ls/R

        if lwR < 0.:
            raise ValueError("Line width / Radius must be greater than 0.")
        elif 0 <= lwR < 1.:
            eta_e = 0.25*lwR**2 + 0.5
        elif 1. <= lwR <= 2.:
            eta_e = -0.25*lwR**2 + lwR

        if lsR < 0.:
            raise ValueError("Line space / Radius must be greater than 0.")
        elif 0 <= lsR < 1.:
            eta_d = 0.5 - 0.25*lsR**2
        elif 1. <= lsR <= 2.:
            eta_d = 1. + 0.25*lsR**2 - lsR

        return eta_e, eta_d

    def _fd_grad(self, img, h=None):
        h = self.metamaterial.resolution[0] if h is None else h
        if self.ops.filt.distance_method == 'periodic':
            # use jnp.roll instead of jnp.gradient b/c periodic boundary conditions
            # right_neighbors  = jnp.roll(img, -1, axis=1)
            # left_neighbors   = jnp.roll(img, 1, axis=1)
            # top_neighbors    = jnp.roll(img, -1, axis=0)
            # bottom_neighbors = jnp.roll(img, 1, axis=0)

            # grad_x = (right_neighbors - left_neighbors) / 2. / h
            # grad_y = (top_neighbors - bottom_neighbors) / 2. / h

            # Compute neighbors using periodic boundary conditions with jnp.roll
            right1 = jnp.roll(img, -1, axis=1)
            left1 = jnp.roll(img, 1, axis=1)
            right2 = jnp.roll(img, -2, axis=1)
            left2 = jnp.roll(img, 2, axis=1)

            top1 = jnp.roll(img, -1, axis=0)
            bottom1 = jnp.roll(img, 1, axis=0)
            top2 = jnp.roll(img, -2, axis=0)
            bottom2 = jnp.roll(img, 2, axis=0)

            # Compute fourth-order central differences
            grad_x = (-right2 + 8*right1 - 8*left1 + left2) / (12 * h)
            grad_y = (-top2 + 8*top1 - 8*bottom1 + bottom2) / (12 * h)

        else:  # assume non-periodicity
            grad_y, grad_x = jnp.gradient(img, h)

        # match return format of jnp.gradient
        return grad_y, grad_x


class OffDiagonalConstraint(VectorConstraint):

    def __init__(self, v, **kwargs):
        super().__init__(**kwargs)
        self.v = v
        self.constraints = [self.g1]
        self.bounds = [lambda t: self.eps * t]
        self.dbdt = np.ones(self.n_constraints) * self.eps

    def g1(self, C):
        m = jnp.diag(jnp.array([1., 1., np.sqrt(2)]))
        C = m @ C @ m
        C /= jnp.linalg.norm(C, ord=2)
        vCv = self.v.T @ C @ self.v
        return jnp.linalg.norm(vCv - jnp.diag(jnp.diag(vCv)))

    def __str__(self):
        return "OffDiagonalConstraint"


class MaterialSymmetryConstraints(VectorConstraint):
    """
    Enforcing material symmetry can be done in a number of ways.
    For example, Andreassen et al. used one scalar constraint on the whole homogenized matrix, but here we implement each constraint as a separate function.

    The terminology and constraints are based on Trageser and Seleson paper.
    https://csmd.ornl.gov/highlight/anisotropic-two-dimensional-plane-strain-and-plane-stress-models-classical-linear
    """

    def __init__(self, symmetry_order='oblique', **kwargs):
        super().__init__(**kwargs)
        self.symmetry_order = symmetry_order

        self.symmetry_types_ = ['oblique',
                                'rectangular', 'square', 'isotropic']

        if self.symmetry_order not in self.symmetry_types_:
            raise ValueError(
                f"Material symmetry must be one of {self.symmetry_types_}")

        if self.symmetry_order in self.symmetry_types_[1:]:
            self.constraints.extend([lambda C: jnp.log((C[0, 2]/C[0, 0])**2/self.eps),
                                     lambda C: jnp.log((C[1, 2]/C[1, 1])**2/self.eps)])
        if self.symmetry_order in self.symmetry_types_[2:]:
            self.constraints.append(lambda C: jnp.log((1. - C[1, 1]/C[0, 0])**2)/self.eps)
        if self.symmetry_order == 'isotropic':
            self.constraints.append(lambda C: jnp.log((1. - C[0, 1]/C[0, 0] - C[2, 2]/C[0, 0])**2)/self.eps)

        self.bounds = self.n_constraints * [lambda t: t]
        self.dbdt = np.ones(self.n_constraints)

    def __str__(self):
        return f"MaterialSymmetryConstraints_{self.symmetry_order}"

class EpigraphBulkModulusConstraint:

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

        # g = lambda C: -0.5 * (C[0][0] + C[1][0])
        def g(C):
            S = jnp.linalg.inv(C)
            return -1. / (S[0][0] + S[0][1]) / 2.
        c, dc_dChom = jax.value_and_grad(g)(jnp.asarray(Chom))

        if grad.size > 0:
            grad[:-1] = dxfem_dx_vjp(np.asarray(dc_dChom).flatten()
                                     @ dChom_dxfem)[0]
            grad[-1] = 0.

        if self.verbose == True:
            print(
                f"- Bulk Modulus: {-c:.2e} (Target ≥{self.aK:.2e}) [{'Satisfied' if -c >= self.aK else 'Not Satisfied'}]")
# ≤

        return self.aK + float(c)

    def compute_K(self, E, nu):
        # computes plane stress bulk modulus from E and nu
        K = E / (3 * (1 - 2 * nu))
        G = E / (2 * (1 + nu))
        K_plane = 9.*K*G / (3.*K + 4.*G)
        return K_plane

        
class VolumeConstraint:
    
    def __init__(self, ops, bound, verbose=True):
        self.ops = ops
        self.bound = bound
        self.verbose = verbose
        
    def __call__(self, x, grad):
        
        x_fem, dxfem_dx_vjp = self.ops.x_fem, self.ops.dxfem_dx_vjp
        
        g, dg_dx_fem = jax.value_and_grad(jnp.mean)(x_fem)
        
        if grad.size > 0:
            grad[:-1] = dxfem_dx_vjp(dg_dx_fem)[0]
            grad[-1] = 0.
        
        if self.verbose:
            print(f"Volume: {g:.3f} (Target <= {self.bound:.3f})")
        
        return float(g - self.bound)

    def __str__(self):
        return "VolumeConstraint"