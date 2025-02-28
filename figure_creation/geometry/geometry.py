import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon
from scipy.spatial import Voronoi, voronoi_plot_2d


class SquareDomain(RegularPolygon):
    """
    A square domain represented as a matplotlib RegularPolygon.

    This class extends RegularPolygon to create squares with additional
    functionality for geometric operations and point containment testing.
    """

    def __init__(self, xy, side_length, orientation=0, **kwargs):
        # For a square, the radius (distance from center to vertex)
        # is related to side length by: radius = side_length / sqrt(2)
        radius = side_length / np.sqrt(2)

        # Initialize as a 4-vertex regular polygon (square)
        super().__init__(
            xy=xy,
            numVertices=4,
            radius=radius,
            orientation=orientation,
            **kwargs
        )

        self.side_length = side_length
        self.half_side = side_length / 2.
        self.grid = None

    def translate(self, v):
        """Move the square by vector v."""
        self.xy += np.asarray(v)

    def rotate(self, angle):
        """Rotate the square by angle (in radians)."""
        self.orientation += angle

    @property
    def bounds(self):
        """Return the axis-aligned bounding box (xmin, xmax, ymin, ymax)."""
        xy = self.xy
        r = self.radius  # Distance from center to corner
        return (xy[0]-r, xy[0]+r, xy[1]-r, xy[1]+r)

    def contains(self, pt):
        """
        Test if a point is inside the square.

        Parameters:
        pt: array-like, shape (2,) - The point coordinates to test

        Returns:
        bool: True if the point is inside the square, False otherwise
        """
        # Convert to numpy arrays
        pt = np.asarray(pt)

        # Translate point to origin-centered coordinates
        centered_pt = pt - self.xy

        # RegularPolygon in matplotlib is rotated 45° by default for squares
        # We need to account for this when checking containment
        correction_factor = np.pi/4

        # Create rotation matrix to undo the square's rotation
        angle = -self.orientation + correction_factor
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[c, -s], [s, c]])

        # Transform the point to the square's local coordinate system
        unrotated_pt = rot_mat @ centered_pt

        # In the local coordinate system, the square is axis-aligned
        # with sides of length side_length
        return (abs(unrotated_pt[0]) <= self.half_side and
                abs(unrotated_pt[1]) <= self.half_side)


class Grid:
    """
    A 2D grid of points with support for rotation and transformation.

    This class represents a rectangular grid of points that can be rotated
    and filtered by domains.
    """

    def __init__(self, x_bounds, y_bounds, dx, dy, angle=0.0, domains=None):
        """
        Initialize a grid with the given parameters.

        Parameters:
        x_bounds: tuple (xmin, xmax) - The range of x coordinates
        y_bounds: tuple (ymin, ymax) - The range of y coordinates
        dx, dy: float - The spacing between grid points
        angle: float - Rotation angle in radians
        domains: list - Optional list of domains to filter points
        """
        # Convert bounds to numpy arrays for consistency
        self.x_bounds = np.asarray(x_bounds)
        self.y_bounds = np.asarray(y_bounds)
        self.dx, self.dy = dx, dy
        self.angle = angle
        self.domains = domains if domains is not None else []

        # Create a meshgrid of points
        xx, yy = np.meshgrid(
            np.arange(x_bounds[0], x_bounds[1] + dx, dx),
            np.arange(y_bounds[0], y_bounds[1] + dy, dy)
        )

        # Flatten the meshgrid to 1D arrays
        self.xx = xx.ravel()
        self.yy = yy.ravel()

        # Apply rotation to the grid points
        rotated_points = self.points @ self.rot_mat.T

        # Update the point coordinates with rotated values
        self.xx = rotated_points[:, 0]
        self.yy = rotated_points[:, 1]

    def add_to_axes(self, ax, **kwargs):
        """Add grid points to a matplotlib axis."""
        ax.scatter(self.xx, self.yy, **kwargs)

    @property
    def rot_mat(self):
        """Return the 2D rotation matrix for the current angle."""
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array([
            [c, -s],
            [s, c]
        ])

    @property
    def points(self):
        """Return grid points as a 2D array of shape (n, 2)."""
        return np.column_stack((self.xx, self.yy))

    @points.setter
    def points(self, new_points):
        """
        Set the grid points directly.

        Parameters:
        new_points: array of shape (n, 2) containing x and y coordinates
        """
        new_points = np.asarray(new_points)
        if new_points.ndim != 2 or new_points.shape[1] != 2:
            raise ValueError(
                f"Expected points with shape (n, 2), got {new_points.shape}")

        self.xx = new_points[:, 0]
        self.yy = new_points[:, 1]
