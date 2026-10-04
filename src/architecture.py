# inputs are x and t
# outputs are B(x,t) and u(x,t), where B is magnetic field and u is velocity field

# Latin Hypercube Sampling for generating collocation points on boundary
# interior points, initial conditions, boundary conditions

# format tensors for automatic differentiation

# Heaviside-Lorentz units? maybe

import jax
import jax.numpy as jnp

def PINN_Architecture(hidden_layers, PRNG_key, activation_function, input_dim=2, output_dim=7):
    # activation function is tanh
    # input dim is 2 because just x and t
    # output dim is 7 because density, 3-velocity, pressure, By, Bz altogether 7 (Bx held constant)

    layer_chain = []
    layer_chain.append(input_dim)
    layer_chain.extend(hidden_layers) # hidden layers is meant to be a list of layers
    layer_chain.append(output_dim)

    params = []

    for size_in, size_out in zip(layer_chain[:-1], layer_chain[1:]):
        # using input key to generate new keys
        PRNG_key, subkey = jax.random.split(PRNG_key)

        # declaring initialization function for weights
        init_fn = jax.nn.initializers.he_normal()
        
        # generating weights with random keys, using PRNG key
        W = init_fn(subkey, shape=(size_in, size_out))

        # biases initialized to zero
        b = jnp.zeros(size_out)

        # storing matrix + bias as tuple in params
        params.append((W, b))

    return params


def call_PINN_Architecture(params, input_array, activation_function):
    # looping over every (matrix,bias) tuple in params
    current_value = input_array
    for (W, b) in params[:-1]:
        current_value = activation_function((current_value @ W) + b)

    # doing last tuple separately from activation function
    W,b = params[-1]
    current_value = (current_value @ W) + b

    return current_value
    


    
