# inputs are x and t
# outputs are B(x,t) and u(x,t), where B is magnetic field and u is velocity field

# Latin Hypercube Sampling for generating collocation points on boundary
# interior points, initial conditions, boundary conditions

# format tensors for automatic differentiation

# this is what a training loop in JAX looks like
'''
for step in range(num_steps):
    grads = jax.grad(loss_fn)(params, x, y)
    params = jax.tree.map(lambda p, g: p - lr*g, params, grads)
'''

# µ_0, ϵ_0 and c set to 1 here - Heaviside-Lorentz units

import jax
import jax.numpy as jnp

class PINNArchitecture:
    def __init__(self, x, t, u_field, B_field):
        self.x = x
        self.t = t

    def u(x,t):
        # boundary conditions for u-field
        if x==0 and t==0:
            u_field = 0
        if t==500:
            u_field = 0.5

    def B(x,t):
        # boundary conditions for B-field
        if x==1000 and t==1000:
            B_field = 0
    
    def get_x(self):
        return x
    def get_t(self):
        return t

    def set_x(self, new_x):
        x = new_x
    def set_t(self, new_t):
        t = new_t

    
    
