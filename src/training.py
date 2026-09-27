# dedicated to training the PINN
# create instance of model, pass vector (input) to instance, observe output, construct loss
# call loss: pass input values to loss, collect output (call function, we give inputs, it gives outputs)
# backpropagation: when implementing architecture, how update weights? grad
# start with random weights, backprop adjusts to actual values
# optimized model --> PINN should act like a solver (first use inputs and BC's)
# validation mechanism - how much it captures physics (or what we need to add); validate for points we don't see
# afterwards plot results with matplotlib
# output is u(x,t) and B(x,t), input 1000x1 vectors for x and t individually

# load model, ask for input/output (accepted shapes)
