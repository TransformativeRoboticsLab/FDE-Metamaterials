import sys
import unittest

import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
from boundaries import PeriodicDomain
from loguru import logger

from metatop.Metamaterial import setup_metamaterial

logger.remove()
logger.add(sys.stderr, level='INFO')


base_config = dict(
    E_max=1.,
    E_min=1/30.,
    nu=0.4,
    nelx=50,
    nely=50,
)


class TestMetamaterialPBC(unittest.TestCase):

    def test_periodic_fluctuation_field(self):
        """
        Tests if the computed fluctuation fields satisfy periodic BCs
        by comparing values on opposite boundaries.
        """
        config = base_config.copy()
        try:
            metamate = setup_metamaterial(**config)
        except Exception as e:
            self.fail(f"setup_metamaterial failed: {e}")
            raise e

        # Initialize if needed (might be handled within solve)
        metamate.x.vector()[:] = np.random.uniform(0, 1, size=metamate.R.dim())

        try:
            sols, Chom = metamate.solve()  # Get fluctuation fields
        except Exception as e:
            self.fail(f"metamate.solve() failed: {e}")
            raise e
        # convert to functions
        sols = [fe.project(sol, metamate.V) for sol in sols]
        self.assertIsNotNone(sols, "Solve method did not return solutions.")
        self.assertIsInstance(sols, (list, tuple),
                              "Solutions should be a list or tuple.")
        self.assertGreater(len(sols), 0, "No solutions returned.")
        self.assertTrue(all(isinstance(s, fe.Function) for s in sols),
                        "Items in sols are not all fe.Functions.")

        # Get domain boundaries (assuming they are stored or accessible from mesh)
        # If not stored on metamate, get from mesh:
        coords = metamate.mesh.coordinates()
        xmin, ymin = coords.min(axis=0)[:2]  # Assumes 2D
        xmax, ymax = coords.max(axis=0)[:2]  # Assumes 2D

        eval_tol = 1e-8  # Tolerance for comparing function values

        # Loop through each solution field (each load case)
        for i, u_prime in enumerate(sols):
            logger.info(f"Checking Periodicity for Solution {i+1}")
            dim = u_prime.geometric_dimension()  # 2D or 3D?
            val_dim = u_prime.value_dimension(0)  # Vector dimension

            # Define sample points (avoid corners for boundary checks)
            num_points = 5
            test_points_y = np.linspace(
                ymin + 0.1*(ymax-ymin), ymax - 0.1*(ymax-ymin), num_points)
            test_points_x = np.linspace(
                xmin + 0.1*(xmax-xmin), xmax - 0.1*(xmax-xmin), num_points)

            # Check Left <-> Right boundary
            logger.info(
                f"Checking Left ({xmin=}) <-> Right ({xmax=}) boundary...")
            for y_coord in test_points_y:
                point_left = fe.Point(xmin, y_coord)
                point_right = fe.Point(xmax, y_coord)
                try:
                    val_left = u_prime(point_left)
                    val_right = u_prime(point_right)
                except RuntimeError as e:
                    self.fail(f"Evaluation failed at y={y_coord}: {e}")

                # Compare each component of the vector field
                for comp in range(val_dim):
                    logger.debug(
                        f"    y={y_coord:.3f}, Comp {comp}: u'({xmin:.1f})={val_left[comp]:.6e}, u'({xmax:.1f})={val_right[comp]:.6e}")
                    self.assertTrue(fe.near(val_left[comp], val_right[comp], eval_tol),
                                    f"Sol {i+1}, Comp {comp} mismatch at y={y_coord}: Left={val_left[comp]}, Right={val_right[comp]}")
            logger.success("Left <-> Right values are consistent")

            # Check Bottom <-> Top boundary
            logger.info(
                f"Checking Bottom ({ymin=}) <-> Top ({ymax=}) boundary...")
            for x_coord in test_points_x:
                point_bottom = fe.Point(x_coord, ymin)
                point_top = fe.Point(x_coord, ymax)
                try:
                    val_bottom = u_prime(point_bottom)
                    val_top = u_prime(point_top)
                except RuntimeError as e:
                    self.fail(f"Evaluation failed at x={x_coord}: {e}")

                # Compare each component
                for comp in range(val_dim):
                    logger.debug(
                        f"    x={x_coord:.3f}, Comp {comp}: u'({ymin:.1f})={val_bottom[comp]:.6e}, u'({ymax:.1f})={val_top[comp]:.6e}")
                    self.assertTrue(fe.near(val_bottom[comp], val_top[comp], eval_tol),
                                    f"Sol {i+1}, Comp {comp} mismatch at x={x_coord}: Bottom={val_bottom[comp]}, Top={val_top[comp]}")
            logger.success("Bottom <-> Top values are consistent")

            # Check Corner Point Equivalence
            logger.info("Checking Corner Equivalence...")
            try:
                val_bl = u_prime(fe.Point(xmin, ymin))
                val_br = u_prime(fe.Point(xmax, ymin))
                val_tl = u_prime(fe.Point(xmin, ymax))
                val_tr = u_prime(fe.Point(xmax, ymax))
            except RuntimeError as e:
                self.fail(f"Corner evaluation failed: {e}")

            logger.debug(
                f"    Values (BL, BR, TL, TR): {val_bl}, {val_br}, {val_tl}, {val_tr}")
            for comp in range(val_dim):
                self.assertTrue(fe.near(
                    val_bl[comp], val_br[comp], eval_tol), f"Sol {i+1}, Comp {comp}: BL!=BR")
                self.assertTrue(fe.near(
                    val_bl[comp], val_tl[comp], eval_tol), f"Sol {i+1}, Comp {comp}: BL!=TL")
                self.assertTrue(fe.near(
                    val_bl[comp], val_tr[comp], eval_tol), f"Sol {i+1}, Comp {comp}: BL!=TR")
            logger.success("Corner values are consistent.")

        logger.success("PASS: Metamaterial solution periodicity")

    def test_periodic_mapping(self):
        """Test the PeriodicDomain mapping functionality."""
        logger.info("Testing PeriodicDomain class mapping...")

        # Create a simple rectangular mesh
        mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(2, 1), 20, 10)

        # Initialize the periodic domain
        periodic_domain = PeriodicDomain(mesh)

        logger.debug(f"Mesh dimensions: x=[{periodic_domain._xmin}, {periodic_domain._xmax}], "
                     f"y=[{periodic_domain._ymin}, {periodic_domain._ymax}]")

        # Test map method for various boundary points
        logger.info("Testing map method for periodic boundary conditions...")

        # Test cases for mapping: source point -> expected mapping
        test_cases = [
            # Right edge points map to left edge
            ([2.0, 0.25], [0.0, 0.25]),
            ([2.0, 0.5], [0.0, 0.5]),
            ([2.0, 0.75], [0.0, 0.75]),

            # Top edge points map to bottom edge
            ([0.5, 1.0], [0.5, 0.0]),
            ([1.0, 1.0], [1.0, 0.0]),
            ([1.5, 1.0], [1.5, 0.0]),

            # Top-right corner maps to bottom-left corner
            ([2.0, 1.0], [0.0, 0.0])
        ]

        for source, expected in test_cases:
            mapped = [0.0, 0.0]  # Initialize target array
            periodic_domain.map(source, mapped)
            logger.debug(f"Mapping {source} -> {mapped} (expected {expected})")
            self.assertTrue(np.allclose(mapped, expected, atol=1e-10),
                            f"Expected {source} to map to {expected}, got {mapped}")

        # Test the inside method for boundary identification
        logger.info("Testing inside method for boundary identification...")

        # Test cases for inside method: point -> expected result
        inside_test_cases = [
            # Points on left boundary (should be inside)
            ([0.0, 0.25], True),
            ([0.0, 0.5], True),
            ([0.0, 0.75], True),

            # Points on bottom boundary (should be inside)
            ([0.5, 0.0], True),
            ([1.0, 0.0], True),
            ([1.5, 0.0], True),

            # Corner points (should be outside per implementation)
            ([0.0, 0.0], True),  # Bottom-left is not explicitly excluded
            ([2.0, 0.0], False),  # Bottom-right is excluded
            ([0.0, 1.0], False),  # Top-left is excluded
            ([2.0, 1.0], False),  # Top-right (not on left or bottom)

            # Other boundary points (should be outside)
            ([2.0, 0.5], False),  # Right edge
            ([1.0, 1.0], False),  # Top edge
        ]

        for point, expected in inside_test_cases:
            result = periodic_domain.inside(point, True)
            logger.debug(
                f"Point {point} inside={result} (expected {expected})")
            self.assertEqual(result, expected,
                             f"Point {point} should be {expected} but got {result}")

        logger.success("PASS: PBC mapping")


# --- Running the Test ---
if __name__ == '__main__':
    logger.info("Running Metamaterial Periodicity Test...")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMetamaterialPBC)
    runner = unittest.TextTestRunner()
    results = runner.run(suite)
