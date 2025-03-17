import jax.numpy as jnp
from jax import grad


def max_function(x):
    return jnp.max(x)


# Define arrays to test
arrays_to_test = [
    jnp.array([1.0, 2.0, 3.0]),  # Unique values
    jnp.array([1.0, 3.0, 3.0]),  # Two max values are equal
    jnp.array([3.0, 3.0, 3.0]),  # All values are equal
    jnp.array([-1.0, -3.0, -2.0]),  # Negative values
]

# Compute gradients
for arr in arrays_to_test:
    grad_max = grad(max_function)(arr)
    print(f"Array: {arr}")
    print(f"Gradient: {grad_max}\n")
