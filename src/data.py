# data is boundary conditions, and how to define x and t
# use this to set boundary conditions, input dimensions

# at production level code, we give user the BC, but for training, we set our own BC

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scipy.stats import qmc

def load_solver_csv(csv_path):
  # for later, not now

def generate_collocation_points(domain_bounds, n_points, key):
  # Latin Hypercube Sampling; integer seed for now, will change later if needed
  LHS_sampler = qmc.LatinHypercube(d=2, seed=0)
  array_to_scale = LHS_sampler.random(n=n_points)
  # bound inputs depends on how domain_bounds is formatted
  scaled_array = qmc.scale(array_to_scale, l_bounds=[0,0], u_bounds=[1000,1000])
  return jnp.asarray(scaled_array)
  

def load_boundary_conditions(bc_input=None):
  #something

