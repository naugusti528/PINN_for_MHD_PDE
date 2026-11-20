# this is where the loss function(s)/mechanisms will go
# (line 58 in equinox jax link is the pde loss)
# for now 1D assuming isothermal, incompressible, and momentum conservation
import jax
import jax.numpy as jnp
from architecture import PINN_Architecture
from data import generate_collocation_points
from constants import B0, ETA, NU, L

B0 = 100 # in units of Gauss
ETA = 10**-3
NU = 10**-4
L = 1000
T = 1000

def B_fn(model, x, t):
 x_t_stacked = jnp.stack([x,t])
 output = model(x_t_stacked)
 return output[0]

def u_fn(model, x, t):
 x_t_stacked = jnp.stack([x,t])
 output = model(x_t_stacked)
 return output[1]

# ---- derivatives of the network ----
u_t = jax.grad(u_fn, argnums=2)
u_x = jax.grad(u_fn, argnums=1)
u_xx = jax.grad(u_x, argnums=1)
 
B_t = jax.grad(B_fn, argnums=2)
B_x = jax.grad(B_fn, argnums=1)
B_xx = jax.grad(B_x, argnums=1)

def pde_residual(model, x, t):
 r_u = u_t(model, x, t) - B0 * B_x(model, x, t) - NU * u_xx(model, x, t)
 r_B = B_t(model, x, t) - B0 * u_x(model, x, t) - ETA * B_xx(model, x, t)
 return r_u, r_B

def residual_loss(model, points):
 points = generate_collocation_points([[0, L], [0, T]], n_points=2000)
 x,t = points[:,0], points[:,1]
 batched_residual = jax.vmap(pde_residual, in_axes=(None,0,0))
 r_u,r_B = batched_residual(model, x, t)
 return jnp.mean(r_u**2 + r_B**2)

def 

if __name__ == "__main__":
 key = jax.random.key(0)
 model = PINN_Architecture(key)
 print("tests:")
 print(u_fn(model, 0.5, 0.1), B_fn(model, 0.5, 0.1))
 print(pde_residual(model, 0.5, 0.1))
