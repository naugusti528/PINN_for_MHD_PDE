# this is where the loss function(s)/mechanisms will go
# (line 58 in equinox jax link is the pde loss)
# for now 1D assuming isothermal, incompressible, and momentum conservation
import jax
import jax.numpy as jnp
from architecture import PINN_Architecture
 
## B0, ETA, L, NU -> DEFINE THESE CONSTANTS
# also should have T

B0 = 1000 # in units of Gauss
ETA = 10**-3
NU = 10**-4
L = 4 # dimensionless value - for 4 gravitational radii
T = 5 # represents light crossing the distance L 5 times

## B_fn, u_fn -> GET FROM MODEL
key = jax.random.key(0)
model = PINN_Architecture(key)

def B_fn(model, x, t):
 x_t_stacked = jnp.stack([x,t])
 model(x_t_stacked)
 return model[0]

def u_fn(model, x, t):
 x_t_stacked = jnp.stack([x,t])
 model(x_t_stacked)
 return model[1]

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
