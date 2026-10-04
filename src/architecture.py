# inputs are x and t
# outputs are B(x,t) and u(x,t), where B is magnetic field and u is velocity field

# Latin Hypercube Sampling for generating collocation points on boundary
# interior points, initial conditions, boundary conditions

# format tensors for automatic differentiation

# Heaviside-Lorentz units? maybe

import jax
import jax.numpy as jnp

def PINN_Architecture(PRNG_key, activation_function, input_dim=2, output_dim=7):
    # activation function is tanh
    # input dim is 2 because just x and t
    # output dim is 7 because density, 3-velocity, By, Bz altogether 7 (Bx held constant)

    hidden_layers = [64,64,64] # 3 layers 64 neurons
    
    layer_chain = []
    layer_chain.append(input_dim)
    layer_chain.extend(hidden_layers)
    layer_chain.append(output_dim)

    # sequential function followed by neuron (to make things easier)
    # can define set of neurons in one layer followed by activation function
    # for reference https://docs.jax.dev/en/latest/notebooks/neural_network_with_tfds_data.html
    # "Utility and Loss Functions" <-- should be together, and separate from data loader

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

'''
move below function up into above function (and use references from jax documentation for simplification)
'''
def call_PINN_Architecture(params, input_array, activation_function):
    # looping over every (matrix,bias) tuple in params
    current_value = input_array
    for (W, b) in params[:-1]:
        current_value = activation_function((current_value @ W) + b)

    # doing last tuple separately from activation function
    W,b = params[-1]
    current_value = (current_value @ W) + b

    # each layer should be sequentially connected
    # activation fn part of main architecture
    # purpose of actv fn is to help neural network learn nonlinear patterns

    return current_value
    


    
