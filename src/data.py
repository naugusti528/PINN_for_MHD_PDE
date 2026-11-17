# data is boundary conditions, and how to define x and t
# use this to set boundary conditions, input dimensions

# at production level code, we give user the BC, but for training, we set our own BC

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scipy.stats import qmc

def load_solver_csv(csv_path):
  # for later, not now

def generate_collocation_points(domain_bounds, n_points, seed=0):
  x_bounds, t_bounds = domain_bounds
  l_bounds = [x_bounds[0], t_bounds[0]]
  u_bounds = [x_bounds[1], t_bounds[1]]
  
  LHS_sampler = qmc.LatinHypercube(d=2, seed=seed)
  array_to_scale = LHS_sampler.random(n=n_points)
  scaled_array = qmc.scale(array_to_scale, l_bounds, u_bounds)
  return jnp.asarray(scaled_array)

  # refer to jax documentation for functions simplifying collocation point creation
  

def load_boundary_conditions(bc_input=None):
  #something

