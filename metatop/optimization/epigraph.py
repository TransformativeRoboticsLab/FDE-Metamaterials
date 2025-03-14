import abc
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
from metatop.mechanics import inv_mandelize, mandelize, ray_q
from metatop.profiling import profile_block, profile_function

from .utils import stop_on_nan


class EpigraphOptimizer(nlopt.opt):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.active_constraints = []

    def setup(self):
        print("Setting up optimizer...")
        self.set_min_objective(EpigraphObjective())
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
                    raise ValueError(
                        f"Constraint function {g} seems to take an incorrect number of arguments")
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
        raise NotImplementedError(
            "Equality constraints are not supported for this optimizer")

    def add_equality_mconstraint(self, *args):
        raise NotImplementedError(
            "Equality constraints are not supported for this optimizer")

    @property
    def size(self):
        return self.get_dimension()

    @property
    def n_constraints(self):
        return len(self.active_constraints)


class EpigraphObjective:
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

        if t > 1e10:
            raise ForcedStop(
                "Objective function value is too large. Terminating optimization run.")

        return t


class EpigraphConstraint(abc.ABC):
    """
    Abstract base class for epigraph constraints in minimax optimization problems.

    These constraints reformulate objectives of the form min_x max{f1, f2, f3}
    into the epigraph form with a slack variable t:

    min_{x,t} t
    s.t. f_i(x) - t <= 0 for all i
    """
    @abc.abstractmethod
    def obj(self, x, C):
        """
        Compute the constraint objective function.

        Parameters:
        - x: Design variables
        - C: Homogenized constitutive tensor

        Returns:
        - c: Constraint values
        - cs: Auxiliary output (e.g., actual Rayleigh quotients)
        """
        pass

    @abc.abstractmethod
    def adjoint(self, x, grad, dxfem_dx_vjp, Chom, dChom_dxfem):
        """
        Compute the adjoint pass for gradients.

        Parameters:
        - x: Design variables
        - grad: Array to store the gradient values
        - dxfem_dx_vjp: VJP function from forward pass
        - Chom: Homogenized constitutive tensor
        - dChom_dxfem: Derivative of Chom w.r.t. xfem
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        return "EpigraphConstraint"

    @abc.abstractmethod
    def _setup_evals_lines(self):
        """
        Setup the evaluation lines for the optimization progress plot.
        This is specific to each derived class.

        This method should set:
        - self.evals_lines: The plot lines for evaluation values
        """
        pass

    # <=====> Methods with implementations <=====>
    """
    Required: basis_v, ops, metamaterial
    """

    def __init__(self, basis_v, ops, metamaterial, extremal_mode, verbose=True, eps=1., silent=False, plot_interval=25, show_plot=True, img_resolution=(200, 200)):

        # Required
        if not np.allclose(basis_v.T @ basis_v, np.eye(3)):
            raise ValueError(
                f"Input basis is not orthonormal:\n{basis_v}")
        self.basis_v = basis_v
        self.ops = ops
        self.metamaterial = metamaterial
        self.extremal_mode = extremal_mode

        # Optional
        self.verbose = verbose
        self.silent = silent
        self.eps = eps
        self.n_constraints = None  # To be defined by subclasses

        # Plotting
        self.plot_interval = plot_interval
        self.show_plot = show_plot
        self.img_resolution = img_resolution
        self.img_shape = (self.metamaterial.width, self.metamaterial.height)

        # Initialize plotting elements
        self.fig = None
        self.evals_lines = None
        self.epoch_lines = []
        self.last_epoch_plotted = -1

    def __call__(self, results, x, grad, dummy_run=False):
        """
        Evaluate the constraint function and its gradient.

        Parameters:
        - results: Array to store the constraint values
        - x: Input vector (design variables and t)
        - grad: Array to store the gradient values
        - dummy_run: If True, only compute results without side effects

        Returns:
        - No return value; modifies results and grad in-place
        """
        # strip t off leaving all the other DOFs
        x_, t = x[:-1], x[-1]
        dxfem_dx_vjp, Chom, dChom_dxfem = self.forward(x_)

        c, cs = self.obj(x_, Chom)
        results[:] = c - t

        if dummy_run:
            return

        if grad.size > 0:
            self.adjoint(x_, grad, dxfem_dx_vjp, Chom, dChom_dxfem)

        self.update_metrics(t, c, cs)

        if hasattr(self, 'plot_interval') and len(self.ops.evals) % self.plot_interval == 1:
            self.update_plot(x_)

        stop_on_nan(c)

    def forward(self, x):
        """
        Compute the forward pass: filter, project, solve the FE problem.

        Parameters:
        - x: Design variables

        Returns:
        - dxfem_dx_vjp: VJP function for backpropagation
        - Chom: Homogenized constitutive tensor
        - dChom_dxfem: Derivative of Chom w.r.t. xfem
        """
        x_fem, dxfem_dx_vjp = jax.vjp(self.filter_and_project, x)

        self.metamaterial.x.vector()[:] = x_fem
        sols, Chom, _ = self.metamaterial.solve()
        Chom = jnp.asarray(Chom)
        E_max, nu = self.metamaterial.prop.E_max, self.metamaterial.prop.nu
        dChom_dxfem = self.metamaterial.homogenized_C(sols, E_max, nu)[1]

        self.ops.update_state(sols, Chom, dChom_dxfem, dxfem_dx_vjp, x_fem)

        return dxfem_dx_vjp, Chom, dChom_dxfem

    def filter_and_project(self, x):
        """
        Apply filtering and projection to the design variables.

        Parameters:
        - x: Design variables

        Returns:
        - Filtered and projected design variables
        """
        x = self.ops.filt_fn(x)
        x = jax_projection(x, self.ops.beta, self.ops.eta)
        return x

    def update_metrics(self, t, c, cs):
        self.ops.evals.append([t, *c])
        if self.silent:
            return

        if self.verbose:
            print("-" * 30)
            print(
                f"Epoch {self.ops.epoch:d}, Step {len(self.ops.evals):d}, Beta = {self.ops.beta:.1f}, Eta = {self.ops.eta:.1f}")
            print("-" * 30)
            print(f"t: {t:.3e} g_ext(x): {c}")
            print(f"Actual Values: {cs}")
        else:
            print(f"{len(self.ops.evals):04d} --\tt: {t:.3e} \n\tg_ext(x): {c}")

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

    def update_plot(self, x_):
        if self.fig is None:
            self._setup_plots()

        # Still None after setup (could happen if show_plot is False)
        if self.fig is None:
            return

        fields = self._prepare_fields(x_)
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
        x_data = range(1, len(self.ops.evals)+1)
        y_data = np.asarray(self.ops.evals)

        # Make sure we only update as many lines as we have data columns
        for i, line in enumerate(self.evals_lines):
            if i < y_data.shape[1]:
                line.set_data(x_data, y_data[:, i])

        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax2.set_xlim(left=0, right=len(self.ops.evals) + 2)

        # we only want to update epoch lines if there is a new one
        for idx in self.ops.epoch_iter_tracker:
            if idx > self.last_epoch_plotted:
                self.last_epoch_plotted = idx
                self.epoch_lines.append(self.ax2.axvline(x=idx, color='black',
                                                         linestyle='--', alpha=0.5, linewidth=3.))

    def _setup_plots(self):
        """Common plot setup for all derived classes"""
        plt.ion() if self.show_plot else plt.ioff()
        self.fig = plt.figure(figsize=(15, 8))
        grid_spec = gridspec.GridSpec(2, 5, )
        self.ax1 = [plt.subplot(grid_spec[0, 0]),
                    plt.subplot(grid_spec[0, 1]),
                    plt.subplot(grid_spec[0, 2]),
                    plt.subplot(grid_spec[0, 3]),
                    plt.subplot(grid_spec[0, 4]),
                    ]
        self.ax2 = plt.subplot(grid_spec[1, :])
        self.ax2.grid(True)
        self.ax2.set(xlabel='Iterations',
                     ylabel='Function Evaluations',
                     xlim=(0, 10),
                     title='Optimization Progress')

        # Call the derived class's implementation to setup evaluation lines
        self._setup_evals_lines()

        # Make sure the lines are set up
        if self.evals_lines is None:
            raise RuntimeError("_setup_evals_lines must set self.evals_lines")


class ExtremalConstraints(EpigraphConstraint):

    def __init__(self, basis_v, ops, metamaterial, extremal_mode, objective_type, weights=jnp.ones(3), **kwargs):
        super().__init__(basis_v, ops, metamaterial, extremal_mode, **kwargs)

        self.objective_type = objective_type

        self.n_constraints = 2 if 'ratio' in self.objective_type else 3
        self.weights = weights

    def obj(self, x, C):
        """
        Matrix Normalization and Eigenvalue Control:

        We normalize the homogenized material matrix (C) to its spectral norm (so the largest eigenvalue = 1) to simplify eigenvalue magnitude control, aiming to align basis vectors (v) with eigenvectors.
        This normalization, while aiding eigenvector alignment, obscures the material's true stiffness.
        Thus we will need to implement additional constraints to ensure basis_v are eigenvectors of C.
        """

        M = mandelize(C)
        # we use S instead of C for bimode materials
        M = jnp.linalg.inv(M) if self.extremal_mode == 2 else M
        M /= jnorm(M, ord=2)

        # Rayleigh quotients with unit length vectors of V
        V = self.basis_v
        r1, r2, r3 = self.weights*ray_q(M, V)

        if self.objective_type == 'ray':
            return jnp.log(jnp.array([r1, 1.-r2, 1.-r3])), jnp.array([r1, r2, r3,])
        elif self.objective_type == 'ray_sq':
            return (jnp.log(jnp.array([r1**2, (1. - r2**2), (1. - r3**2), ])+1e-8), jnp.array([r1, r2, r3, ]))
        elif self.objective_type == 'uni':
            return (jnp.log(jnp.array[r1, r2 - r1 + 1e-3, r3 - r1 + 1e-3]), jnp.array([r1, r2, r3]))
        # elif self.objective_type == 'norm':
            # return jnp.log(self.w*(jnp.array([jnorm(Cv1), 1-jnorm(Cv2), 1-jnorm(Cv3)])+1e-8)), jnp.array([jnorm(Cv1), jnorm(Cv2), jnorm(Cv3)])
        # elif self.objective_type == 'norm_sq':
            # return (jnp.log(self.w*jnp.array([Cv1@Cv1, 1 - Cv2@Cv2, 1-Cv3@Cv3, ])+1e-8), jnp.array([Cv1@Cv1, Cv2@Cv2, Cv3@Cv3]))
        elif self.objective_type == 'ratio':
            return jnp.log(jnp.array([r1/r2, r1/r3, ])), jnp.array([r1, r2, r3])
        elif self.objective_type == 'ratio_sq':
            return jnp.log(jnp.array([(r1/r2)**2, (r1/r3)**2, ])), jnp.array([r1, r2, r3])
        elif self.objective_type == 'ratio_c1sq':
            return jnp.log(jnp.array([r1**2/r2, r1**2/r3, ])), jnp.array([r1, r2, r3])
        else:
            raise ValueError(
                f"Objective '{self.objective_type}' type not found.")

    def adjoint(self, x, grad, dxfem_dx_vjp, Chom, dChom_dxfem):
        # argnums=1 because we only care about derivative w.r.t. Chom here
        dc_dChom = jax.jacrev(self.obj,
                              argnums=1,
                              has_aux=True)(x, Chom)[0].reshape((self.n_constraints, 9))
        for n in range(self.n_constraints):
            grad[n, :-1] = dxfem_dx_vjp(dc_dChom[n, :] @ dChom_dxfem)[0]
            grad[n, -1] = -1.

    def _setup_evals_lines(self):
        d = np.ones(self.n_constraints+1)
        o = self.objective_type
        if 'ratio' == o:
            labels = [r'$t$',
                      r'$(v_1^T C v_1)/(v_2^T C v_2)$',
                      r'$(v_1^T C v_1)/(v_3^T C v_3)$']
        elif 'ray' == o:
            labels = [r'$t$',
                      r'$(v_1^T C v_1)$',
                      r'$1-(v_2^T C v_2)$',
                      r'$1-(v_3^T C v_3)$']
        elif 'ray_sq' == o:
            labels = [r'$t$',
                      r'$(v_1^T C v_1)^2$',
                      r'$1-(v_2^T C v_2)^2$',
                      r'$1-(v_3^T C v_3)^2$']
        else:
            labels = ['$t$',
                      *(f'{o}_{n}' for n in range(self.n_constraints))]
        """Setup the evaluation lines specific to ExtremalConstraints"""
        if 'ratio' in self.objective_type:
            self.evals_lines = self.ax2.plot([d],
                                             [d],
                                             marker='.',
                                             label=labels,
                                             )
        else:
            self.evals_lines = self.ax2.plot([d],
                                             [d],
                                             marker='.',
                                             label=labels,
                                             )
        self.ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    def __str__(self):
        return "ExtremalConstraints"


class EigenvalueProblemConstraints(EpigraphConstraint):
    def __init__(self, basis_v, ops, metamaterial, extremal_mode, weights=jnp.ones(3), check_valid=False, **kwargs):
        # Initialize with base EpigraphConstraint initialization
        super().__init__(basis_v, ops, metamaterial, extremal_mode, **kwargs)

        self.weights = weights
        self.check_valid = check_valid
        self.n_constraints = 7

        # Setup plots same as ExtremalConstraints
        self.img_resolution = (200, 200)
        self.img_shape = (self.metamaterial.width, self.metamaterial.height)

    def obj(self, x_: np.ndarray, C):
        # We grab U out of x_ so ease derivative computation
        return self._obj_wrt_U(self._select_U(x_), C)

    # We only care about the derivative of the objective w.r.t. the specific parts of x that comprise U.
    # So we do this subcall to avoid differentiating against all of x.
    def _obj_wrt_U(self, U, C):
        M = mandelize(C)
        M = jnp.linalg.inv(M) if self.extremal_mode == 2 else M
        M /= jnorm(M, ord=2)

        # Rayleigh quotients with U
        r1, r2, r3 = self.weights*ray_q(M, U)
        rays = jnp.array([r1, 1.-r2, 1.-r3])

        V = self.basis_v
        U_norms = jnp.linalg.norm(U, axis=0)
        V_norms = jnp.linalg.norm(V, axis=0)
        cosines = (1 + 1e-6 - jnp.diag(U.T@V) / (U_norms*V_norms))/self.eps

        ortho = jnp.linalg.norm(U.T @ U - jnp.eye(3), ord='fro')

        return (jnp.log(jnp.hstack([rays, cosines, [ortho]])),
                jnp.hstack([jnp.array([r1, r2, r3]), cosines, [ortho]]))

    def adjoint(self, x_, grad, dxfem_dx_vjp, Chom, dChom_dxfem):
        # argnums=[0,1] because we care about both derivatives
        jac_func = jax.jacrev(self._obj_wrt_U, argnums=[0, 1], has_aux=True)
        dc_dU, dc_dChom = jac_func(self._select_U(x_), Chom)[0]

        # both arguments get passed in a 3x3 matrices so we unfold them for easier referencing
        dc_dU = dc_dU.reshape((self.n_constraints, 9))
        dc_dChom = dc_dChom.reshape((self.n_constraints, 9))

        for n in range(self.n_constraints):
            grad[n, :-10] = dxfem_dx_vjp(dc_dChom[n, :] @ dChom_dxfem)[0]
            grad[n, -10:-1] = dc_dU[n, :]
            grad[n, -1] = -1.

    def forward(self, x_):
        return super().forward(self._strip_U(x_))

    def update_plot(self, x_):
        super().update_plot(self._strip_U(x_))

    def _strip_U(self, x_):
        return x_[:-9]

    def _select_U(self, x_):
        return x_[-9:].reshape((3, 3))

    def _check_validity(self, x_):
        if not self.check_valid:
            return
        if not self._strip_U(x_).size == self.metamaterial.R.dim():
            raise ValueError(f"Mismatching size of x and R.dim()")

    def _setup_evals_lines(self):
        """Setup evaluation lines specific to EigenvalueProblemConstraints"""
        d = np.ones(self.n_constraints+1)
        labels = ['$t$',
                  '$u_1^TCu_1$',
                  '$1-u_2^TCu_2$',
                  '$1-u_3^TCu_3$',
                  '$\cos(u_1,v_1)$',
                  '$\cos(u_2,v_2)$',
                  '$\cos(u_3,v_3)$',
                  'ortho']
        self.evals_lines = self.ax2.plot(
            [d], [d], marker='.',
            label=labels
        )
        self.ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    def __str__(self):
        return "EigenvalueProblemConstraints"


class SpectralNormConstraint:

    def __init__(self, ops, bound=1., verbose=True):
        self.ops = ops
        self.bound = bound
        self.verbose = verbose

    def __call__(self, x, grad, ):

        Chom, dChom_dxfem, dxfem_dx_vjp = self.ops.Chom, self.ops.dChom_dxfem, self.ops.dxfem_dx_vjp

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
                grad[n, :-
                     1] = dxfem_dx_vjp(dg_dChoms[n].flatten() @ dChom_dxfem)[0]
                grad[n, -1] = -self.dbdt[n]

    @property
    def n_constraints(self):
        return len(self.constraints)

    def __str__(self):
        return "MinimaxConstraints"


class EigenvectorConstraint:

    def __init__(self, basis_v, ops, eps=1e-3, verbose=True):
        self.basis_v = basis_v
        self.ops = ops
        self.eps = eps
        self.verbose = verbose

        self.n_constraints = 3

    def __call__(self, results, x, grad, dummy_run=False):

        Chom, dChom_dxfem, dxfem_dx_vjp = self.ops.Chom, self.ops.dChom_dxfem, self.ops.dxfem_dx_vjp

        t = x[-1]

        c, cs = self.obj(Chom)
        stop_on_nan(c)
        results[:] = c - t

        if dummy_run:
            return

        if grad.size > 0:
            dc_dChom = jax.jacrev(self.obj, has_aux=True)(jnp.asarray(Chom))[
                0].reshape((self.n_constraints, 9))
            for n in range(self.n_constraints):
                grad[n, :-1] = dxfem_dx_vjp(dc_dChom[n, :] @ dChom_dxfem)[0]
                grad[n, -1] = -1.
                # grad[n,:] = dxfem_dx_vjp(dc_dChom[n,:] @ dChom_dxfem)[0]

        if self.verbose:
            print(f"Eigenvector Constraint:")
            print(f"Values: {c}")
            print(f"Residuals: {cs}")
        else:
            print(f"\tg_vec(x): {c}")

    def obj(self, C):
        M = mandelize(C)
        # eigenvectors are the same for C and S so we don't worry about inverting like we do in other constraints
        M /= jnorm(M, ord=2)

        V = self.basis_v
        # Rayleigh quotient in diagonal matrix form
        R = jnp.diag(ray_q(M, V))
        # Resdiuals of eigenvector alignment
        res = M @ V - V @ R
        # norm squared of each residual
        norm_sq = jnp.sum(jnp.square(res), axis=0)

        return jnp.log(norm_sq/self.eps), norm_sq

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
            grad[-1] = 0.

        print(f"\tg_trc(x): {-c:.3f} (Target >={self.bound:.3f})")

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
            dc_dChom = jax.jacrev(obj)(jnp.asarray(
                Chom)).reshape((self.n_constraints, 9))
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


class OffDiagonalConstraint:
    def __init__(self, v, ops, eps=1e-3, verbose=True):
        self.v = v
        self.ops = ops
        self.eps = eps
        self.verbose = verbose

        self.n_constraints = 1

    def __call__(self, x, grad):

        Chom, dChom_dxfem, dxfem_dx_vjp = self.ops.Chom, self.ops.dChom_dxfem, self.ops.dxfem_dx_vjp

        c, dc_dChom = jax.value_and_grad(self.obj)(Chom)

        if grad.size > 0:
            grad[:-1] = dxfem_dx_vjp(dc_dChom.flatten() @ dChom_dxfem)[0]

        print(f"\tg_dia(x): {c}")

        return float(c)

    def obj(self, C):
        m = jnp.diag(np.array([1., 1., np.sqrt(2)]))
        C = m @ C @ m
        vCv = self.v.T @ C @ self.v
        return jnp.log((vCv[0, 1]**2 + vCv[0, 2]**2 + vCv[1, 2]**2)/self.eps)
        # return jnp.linalg.norm(vCv - jnp.diag(jnp.diag(vCv)))

    def __str__(self):
        return "OffDiagonalConstraint"

# class OffDiagonalConstraint(VectorConstraint):

#     def __init__(self, v, **kwargs):
#         super().__init__(**kwargs)
#         self.v = v
#         self.constraints = [self.g1]
#         self.bounds = [lambda t: self.eps * t]
#         self.dbdt = np.ones(self.n_constraints) * self.eps

#     def g1(self, C):
#         m = jnp.diag(jnp.array([1., 1., np.sqrt(2)]))
#         C = m @ C @ m
#         # C /= jnp.linalg.norm(C, ord=2)
#         vCv = self.v.T @ C @ self.v
#         return jnp.linalg.norm(vCv - jnp.diag(jnp.diag(vCv)))

#     def __str__(self):
#         return "OffDiagonalConstraint"


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
            self.constraints.append(lambda C: jnp.log(
                (1. - C[1, 1]/C[0, 0])**2)/self.eps)
        if self.symmetry_order == 'isotropic':
            self.constraints.append(lambda C: jnp.log(
                (1. - C[0, 1]/C[0, 0] - C[2, 2]/C[0, 0])**2)/self.eps)

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
