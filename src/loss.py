# this is where the loss function(s)/mechanisms will go

# prevent overfitting:
# dropout + batch norm

# line 58 in equinox jax link is the pde loss

# for now 1D assuming isothermal, incompressible, and momentum conservation

import jax
import jax.numpy as jnp
 
## B0, ETA, L, NU -> DEFINE THESE CONSTANTS
# also should have T

## B_fn, u_fn -> GET FROM MODEL
 
# ---- derivatives of the network ----
u_t = jax.grad(u_fn, argnums=2)
u_x = jax.grad(u_fn, argnums=1)
u_xx = jax.grad(u_x, argnums=1)
 
B_t = jax.grad(B_fn, argnums=2)
B_x = jax.grad(B_fn, argnums=1)
B_xx = jax.grad(B_x, argnums=1)

B0 = 1000 # in units of Gauss
ETA = 10**-3
NU = 10**-4
L = 4 # dimensionless value - for 4 gravitational radii
T = 5 # represents light crossing the distance L 5 times

def pde_residual(model, x, t):
    r_u = u_t(model, x, t) - B0 * B_x(model, x, t) - NU * u_xx(model, x, t)
    r_B = B_t(model, x, t) - B0 * u_x(model, x, t) - ETA * B_xx(model, x, t)
    return r_u, r_B
