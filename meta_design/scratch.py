import jax
import jax.numpy as jnp
import time

# Enable double precision
jax.config.update("jax_enable_x64", True)

# Define a sample computation
def sample_computation(x):
    return jnp.sum(jnp.sin(x) ** 2 + jnp.cos(x) ** 2)

# Create input data
size = 1000000
x = jnp.linspace(0, 100, size)

# Benchmark with double precision
start_time = time.time()
result = sample_computation(x).block_until_ready()
double_precision_time = time.time() - start_time
print(f"Double Precision Time: {double_precision_time:.6f} seconds")

# Disable double precision
jax.config.update("jax_enable_x64", False)

# Benchmark with single precision
start_time = time.time()
result = sample_computation(x).block_until_ready()
single_precision_time = time.time() - start_time
print(f"Single Precision Time: {single_precision_time:.6f} seconds")

# Compare results
print(f"Performance Impact: {double_precision_time / single_precision_time:.2f}x slowdown")
