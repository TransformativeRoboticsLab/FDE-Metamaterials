import fenics as fe
import numpy as np


class Boundary(fe.SubDomain):
    def __init__(self, mesh, tol=None, **kwargs):
        super().__init__(**kwargs)
        self._xmin, self._ymin = mesh.coordinates().min(axis=0)
        self._xmax, self._ymax = mesh.coordinates().max(axis=0)
        # hmin and hmax needed to determine radius for point conditions
        self._hmax, self._hmin = mesh.hmax(), mesh.hmin()
        self._tol = tol or fe.DOLFIN_EPS

        # print(f"x min/max: {self._xmin}, {self._xmax}")
        # print(f"y min/max: {self._ymin}, {self._ymax}")

    def inside(self, x, on_boundary):
        raise NotImplementedError()

    def _on_left(self, x, on_boundary=True):
        return on_boundary and fe.near(x[0], self._xmin, self._tol)

    def _on_right(self, x, on_boundary=True):
        return on_boundary and fe.near(x[0], self._xmax, self._tol)

    def _on_bot(self, x, on_boundary=True):
        return on_boundary and fe.near(x[1], self._ymin, self._tol)

    def _on_top(self, x, on_boundary=True):
        return on_boundary and fe.near(x[1], self._ymax, self._tol)


class Exterior(Boundary):
    def inside(self, x, on_boundary):
        return on_boundary and (
            self._on_left(x, on_boundary) or
            self._on_right(x, on_boundary) or
            self._on_bot(x, on_boundary) or
            self._on_top(x, on_boundary))


class Left(Boundary):
    def inside(self, x, on_boundary):
        return self._on_left(x, on_boundary)


class Right(Boundary):
    def inside(self, x, on_boundary):
        return self._on_right(x, on_boundary)


class Bottom(Boundary):
    def inside(self, x, on_boundary):
        return self._on_bot(x, on_boundary)


class Top(Boundary):
    def inside(self, x, on_boundary):
        return self._on_top(x, on_boundary)

# The corner point boundaries do not call subroutines because for a 'pointwise' DirichletBC method
# the boolean value 'on_boundary' will always return false


class LowerLeft(Boundary):
    def inside(self, x, on_boundary):
        return fe.near(x[0], self._xmin, self._tol) and fe.near(x[1], self._ymin, self._tol)


class LowerRight(Boundary):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[0], self._xmax, self._tol) and fe.near(x[1], self._ymin, self._tol)


class UpperLeft(Boundary):
    def inside(self, x, on_boundary):
        return fe.near(x[0], self._xmin, self._tol) and fe.near(x[1], self._ymax, self._tol)


class UpperRight(Boundary):
    def inside(self, x, on_boundary):
        return fe.near(x[0], self._xmax, self._tol) and fe.near(x[1], self._ymax, self._tol)


class PeriodicDomain(Boundary):
    def inside(self, x, on_boundary):
        if not on_boundary:
            return False

        is_left = self._on_left(x, on_boundary)
        is_bottom = self._on_bot(x, on_boundary)

        is_lower_right = self._on_bot(
            x, on_boundary) and self._on_right(x, on_boundary)
        is_upper_left = self._on_top(
            x, on_boundary) and self._on_left(x, on_boundary)
        is_corner = is_lower_right or is_upper_left

        return (is_left or is_bottom) and (not is_corner)

    def map(self, x, y):
        dx, dy = self._xmax - self._xmin, self._ymax - self._ymin
        # Map Top-Right Corner -> Bottom Left-Corner
        if self._on_top(x) and self._on_right(x):
            y[0] = x[0] - dx
            y[1] = x[1] - dy
        # Map Right Edge -> Left Edge
        elif self._on_right(x):
            y[0] = x[0] - dx
            y[1] = x[1]
        # Map Top Edge -> Bottom Edge
        else:
            y[0] = x[0]
            y[1] = x[1] - dy

# For Dirichlet BC in top and right


class QuarterDomain(Boundary):
    def inside(self, x, on_boundary):
        if not on_boundary:
            return False

        return self._on_right(x) or self._on_top(x)


class Delta(fe.UserExpression):
    def __init__(self, eps=fe.DOLFIN_EPS, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def eval(self, values, x):
        eps = self.eps
        values[0] = eps/np.pi/(np.linalg.norm(x)**2 + eps**2)
        values[1] = 0

    def value_shape(self): return (2, )
