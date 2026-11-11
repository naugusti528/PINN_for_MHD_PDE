# inputs are x and t
# outputs are B(x,t) and u(x,t), where B is magnetic field and u is velocity field

# format tensors for automatic differentiation
# Heaviside-Lorentz units? maybe

# can use any built in features in jax
# hint: define sequential network. input, then next layer (then 1-line activ func), so on;
# 

import jax
import equinox as eqx

def PINN_Architecture(PRNG_key, input_dim=2, output_dim=2):
    # activation function is tanh - apply to last layer
    # layers in between (hidden layers) will use ReLU
    # input dim, output dim = (x,t), (B,u)
    model = eqx.nn.MLP(in_size=input_dim, out_size=output_dim, width_size=64, depth=3,
        activation=jax.nn.relu, final_activation=jax.nn.tanh, key=PRNG_key)

    return model

    # sequential function followed by neuron (to make things easier)
    # can define set of neurons in one layer followed by activation function
    # for reference https://docs.jax.dev/en/latest/notebooks/neural_network_with_tfds_data.html
    # "Utility and Loss Functions" <-- should be together, and separate from data loader

    # rewrite architecture with equinox, initialize with mlp or fcn (fully connected netwokr)
    # define 3 hidden layers (refer to https://github.com/Ceyron/pinns-in-jax/blob/main/poisson_pinn_in_jax_equinox.ipynb)
    # line 48 in the link above, in_size and out_size are vectors, we might need array within array
    # first + middle are ReLU, last layer is tanh. extend a little but not difficult
    # define one layer, put activation function, deter to next layer
    # last layer has out, apply tanh to that
    # then define loss function. in the given link it's line 60
