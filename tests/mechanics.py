import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from loguru import logger

from metatop.mechanics import *

# --- Configuration ---

# Try to enable 64-bit floats in JAX for precision matching NumPy
try:
    jax.config.update("jax_enable_x64", True)
    _HAS_X64 = True
except Exception:
    _HAS_X64 = False
    warnings.warn(
        "Could not enable JAX 64-bit precision (x64). Tests might fail due to precision differences.")

# Configure Loguru
logger.remove()
# Change level to "DEBUG" for more verbose logs
logger.add(sys.stderr, level="INFO")

logger.info(f"JAX using device: {jax.default_backend()}")
logger.info(f"JAX 64-bit precision enabled: {_HAS_X64}")

# Use float64 for better comparison across libraries
TEST_DTYPE = np.float64

# For matrix2tensor functions
V_test_np = np.array([[11.2, 2.5, 0.1],
                      [2.4, 8.8, -0.5],
                      [0.3, -0.4, 3.1]], dtype=TEST_DTYPE)
V_test_jnp = jnp.array(V_test_np)

# For calculate_elastic_constants functions (needs to be invertible)
# Using a known isotropic material example (Aluminum approx) GPa
E_iso = 70.0
nu_iso = 0.33
G_iso = E_iso / (2.0 * (1.0 + nu_iso))
# Plane stress compliance matrix [ [1/E, -nu/E, 0], [-nu/E, 1/E, 0], [0, 0, 1/G] ]
S_iso_ps = np.array([[1/E_iso, -nu_iso/E_iso, 0],
                     [-nu_iso/E_iso, 1/E_iso, 0],
                     [0, 0, 1/G_iso]], dtype=TEST_DTYPE)
M_test_np = np.linalg.inv(S_iso_ps)  # Stiffness matrix C
logger.info(
    f"Using test stiffness matrix M (derived from isotropic E={E_iso}, nu={nu_iso}):\n{M_test_np}")

# Ensure it's invertible for the test
try:
    np.linalg.inv(M_test_np)
except np.linalg.LinAlgError:
    pytest.fail(
        "Test matrix M is singular, cannot run calculate_elastic_constants tests.")

M_test_jnp = jnp.array(M_test_np)


# --- Test Functions ---

# Tolerance settings - might need adjustment based on precision (float32 vs float64)
RTOL = 1e-9 if _HAS_X64 else 1e-5
ATOL = 1e-9 if _HAS_X64 else 1e-5


@pytest.mark.parametrize("input_style", ['standard', 'mandel'])
def test_matrix2tensor_equivalence(input_style):
    """Compares original loop NumPy matrix2tensor with vectorized NumPy version."""
    logger.info(
        f"Test: NumPy loop vs NumPy vectorized matrix2tensor | style='{input_style}'")
    try:
        C_original = matrix2tensor(V_test_np, input_style=input_style)
        C_vectorized = matrix2tensor_vectorized(
            V_test_np, input_style=input_style)
        np.testing.assert_allclose(
            C_original, C_vectorized, rtol=RTOL, atol=ATOL)
        logger.success("Results MATCH")
    except Exception as e:
        logger.exception(f"Test failed")
        pytest.fail(f"NumPy loop vs vectorized mismatch or error: {e}")


@pytest.mark.parametrize("input_style", ['standard', 'mandel'])
def test_matrix2tensor_numpy_vectorized_vs_jax(input_style):
    """Compares vectorized NumPy matrix2tensor with JAX vectorized version."""
    logger.info(
        f"Test: NumPy vectorized vs JAX vectorized matrix2tensor | style='{input_style}'")
    try:
        C_numpy = matrix2tensor_vectorized(V_test_np, input_style=input_style)
        C_jax = matrix2tensor_vectorized_jnp(
            V_test_jnp, input_style=input_style)
        # Convert JAX output to NumPy for comparison
        C_jax_np = np.array(C_jax)
        np.testing.assert_allclose(C_numpy, C_jax_np, rtol=RTOL, atol=ATOL)
        logger.success("Results MATCH")
    except Exception as e:
        logger.exception(f"Test failed")
        pytest.fail(
            f"NumPy vectorized vs JAX vectorized mismatch or error: {e}")


@pytest.mark.parametrize("input_style", ['standard', 'mandel'])
def test_calculate_elastic_constants_numpy_vs_jax(input_style):
    """Compares original NumPy calculate_elastic_constants (using loops) with JAX einsum version (using vectorized)."""
    logger.info(
        f"Test: Original NumPy vs JAX einsum calculate_elastic_constants | style='{input_style}'")
    try:
        props_numpy = calculate_elastic_constants(
            M_test_np, input_style=input_style)
        props_jax = calculate_elastic_constants_jnp_einsum(
            M_test_jnp, input_style=input_style)
        # Convert JAX dict values
        props_jax_np = {k: np.array(v) for k, v in props_jax.items()}

        assert props_numpy.keys() == props_jax_np.keys(), "Output dict keys differ"
        for key in props_numpy:
            logger.debug(f"Comparing key: {key}")
            np.testing.assert_allclose(
                props_numpy[key], props_jax_np[key], rtol=RTOL, atol=ATOL
            )
        logger.success("Results MATCH")
    except Exception as e:
        logger.exception(
            f"Test failed for key '{key if 'key' in locals() else 'N/A'}' or dict structure")
        pytest.fail(
            f"Original NumPy vs JAX einsum calculate_elastic_constants mismatch or error: {e}")


@pytest.mark.parametrize("input_style", ['standard', 'mandel'])
def test_calculate_elastic_constants_jax_consistency(input_style):
    """Compares JAX einsum vs JAX no_einsum versions of calculate_elastic_constants."""
    logger.info(
        f"Test: JAX einsum vs JAX no_einsum calculate_elastic_constants | style='{input_style}'")
    try:
        props_jax_einsum = calculate_elastic_constants_jnp_einsum(
            M_test_jnp, input_style=input_style)
        props_jax_no_einsum = calculate_elastic_constants_jnp_no_einsum(
            M_test_jnp, input_style=input_style)

        # Convert both to NumPy dicts for comparison
        props_jax_einsum_np = {k: np.array(v)
                               for k, v in props_jax_einsum.items()}
        props_jax_no_einsum_np = {k: np.array(
            v) for k, v in props_jax_no_einsum.items()}

        assert props_jax_einsum_np.keys(
        ) == props_jax_no_einsum_np.keys(), "Output dict keys differ"
        for key in props_jax_einsum_np:
            logger.debug(f"Comparing key: {key}")
            np.testing.assert_allclose(
                props_jax_einsum_np[key], props_jax_no_einsum_np[key], rtol=RTOL, atol=ATOL
            )
        logger.success("Results MATCH")
    except Exception as e:
        logger.exception(
            f"Test failed for key '{key if 'key' in locals() else 'N/A'}' or dict structure")
        pytest.fail(
            f"JAX einsum vs JAX no_einsum calculate_elastic_constants mismatch or error: {e}")


@pytest.mark.parametrize("input_style", ['standard', 'mandel'])
def test_calculate_elastic_constants_isotropic_values(input_style):
    """Checks calculate_elastic_constants output against known isotropic values."""
    logger.info(
        f"Test: Isotropic Validation | style='{input_style}' | E={E_iso}, nu={nu_iso}")

    # Expected results for the isotropic case defined at top of file
    expected_results = {
        'E1': E_iso,
        'E2': E_iso,
        'G12': G_iso,  # Expected G = E / (2*(1+nu)) approx 26.3158
        'nu12': nu_iso,
        'nu21': nu_iso,
        'eta121': 0.0,
        'eta122': 0.0
    }

    try:
        # Use the M_test_np stiffness matrix already defined in the file
        # Assumes calculate_elastic_constants takes stiffness matrix M
        results_numpy = calculate_elastic_constants(
            M_test_np, input_style=input_style
        )

        assert results_numpy.keys() == expected_results.keys(), \
            f"Output keys mismatch for style='{input_style}'"

        mismatches = []
        for key in expected_results:
            expected = expected_results[key]
            actual = results_numpy[key]
            logger.debug(
                f"Comparing key: {key} | Expected: {expected:.6f} | Actual: {actual:.6f}")
            if not np.isclose(actual, expected, rtol=RTOL, atol=ATOL):
                mismatches.append(
                    f"{key} (Exp: {expected:.6f}, Got: {actual:.6f})")

        if mismatches:
            pytest.fail(
                f"Isotropic test mismatch for style='{input_style}': {'; '.join(mismatches)}")
        else:
            logger.success(f"Isotropic test PASSED for style='{input_style}'")

    except Exception as e:
        logger.exception(f"Test failed for style='{input_style}'")
        pytest.fail(
            f"Isotropic test failed for style='{input_style}': {e}")
