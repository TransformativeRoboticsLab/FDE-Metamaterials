import jax
import jax.numpy as jnp

def compute_Ciso(Cstar):
    Ciso = jnp.zeros_like(Cstar)
    Ciso = Ciso.at[0, 1].set(Cstar[0, 1])
    Ciso = Ciso.at[1, 0].set(Cstar[1, 0])
    avg = (Cstar[1, 1] + Cstar[2, 2]) / 2
    Ciso = Ciso.at[1, 1].set(avg)
    Ciso = Ciso.at[2, 2].set(avg)
    Ciso = Ciso.at[0, 0].set((Cstar[0, 1] + avg) * 2)
    return Ciso

def g(Cstar):
    Ciso = compute_Ciso(Cstar)
    diff = Ciso - Cstar
    g_val = jnp.sum(diff ** 2) / Ciso[1, 1]
    return g_val

# Example input
Cstar = jnp.array([[1.0, 0.5, 0.0],
                   [0.5, 1.5, 0.0],
                   [0.0, 0.0, 2.0]])

# Compute the gradient
grad_g = jax.grad(g)(Cstar)
print("Gradient of g with respect to Cstar:\n", grad_g)

# Test for compute_Ciso function
def test_compute_Ciso():
    Cstar_test = jnp.array([[1.0, 0.5, 0.0],
                            [0.5, 1.5, 0.0],
                            [0.0, 0.0, 2.0]])

    expected_Ciso = jnp.array([[3.0, 0.5, 0.0],
                               [0.5, 1.75, 0.0],
                               [0.0, 0.0, 1.75]])

    computed_Ciso = compute_Ciso(Cstar_test)
    
    assert jnp.allclose(computed_Ciso, expected_Ciso), f"Expected {expected_Ciso}, but got {computed_Ciso}"

test_compute_Ciso()
