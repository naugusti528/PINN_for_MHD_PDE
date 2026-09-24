# inputs are x and t

# Latin Hypercube Sampling for generating collocation points on boundary
# interior points, initial conditions, boundary conditions

# format tensors for automatic differentiation

# this is what a training loop in JAX looks like
'''
for step in range(num_steps):
    grads = jax.grad(loss_fn)(params, x, y)
    params = jax.tree.map(lambda p, g: p - lr*g, params, grads)
'''
