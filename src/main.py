import jax
import jax.numpy as jnp
from architecture import PINN_Architecture
from loss import u_fn, B_fn, pde_residual

if __name__ == "__main__":
 key = jax.random.key(0)
 model = PINN_Architecture(key)
 print("tests:")
 print(u_fn(model, 0.5, 0.1), B_fn(model, 0.5, 0.1))
 print(pde_residual(model, 0.5, 0.1))
