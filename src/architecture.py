# inputs are x and t
# outputs are B(x,t) and u(x,t), where B is magnetic field and u is velocity field

# Latin Hypercube Sampling for generating collocation points on boundary
# interior points, initial conditions, boundary conditions

# format tensors for automatic differentiation

# Heaviside-Lorentz units? maybe

# can use any built in features in jax

# hint: define sequential network. input, then next layer (then 1-line activ func), so on;
# 

import jax
import jax.numpy as jnp

def PINN_Architecture(PRNG_key, input_array, activation_function, input_dim=2, output_dim=2):
    # activation function is tanh - apply to last layer
    # layers in between (hidden layers) will use ReLU
    # input dim is 2 because just x and t
    # output dim is 2 because 3-velocity, 3-Bfield

    hidden_layers = [64,64,64] # 3 layers 64 neurons
    
    layer_chain = []
    layer_chain.append(input_dim)
    layer_chain.extend(hidden_layers)
    layer_chain.append(output_dim)

    # sequential function followed by neuron (to make things easier)
    # can define set of neurons in one layer followed by activation function
    # for reference https://docs.jax.dev/en/latest/notebooks/neural_network_with_tfds_data.html
    # "Utility and Loss Functions" <-- should be together, and separate from data loader
    # implement equinox

    # rewrite architecture with equinox, initialize with mlp or fcn (fully connected netwokr)
    # define 3 hidden layers (refer to https://github.com/Ceyron/pinns-in-jax/blob/main/poisson_pinn_in_jax_equinox.ipynb)
    # line 48 in the link above, in_size and out_size are vectors, we might need array within array
    # first + middle are ReLU, last layer is tanh. extend a little but not difficult
    # define one layer, put activation function, deter to next layer
    # last layer has out, apply tanh to that
    # then define loss function. in the given link it's line 60

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

    x = input_array
    for (W, b) in params[:-1]:
        current_value = activation_function((x @ W) + b)
        # we don't have to apply to eat neuron

    # doing last tuple separately from activation function
    W,b = params[-1]
    x = (x @ W) + b

    # each layer should be sequentially connected

    return x
