from tf_pde import *
print (tf.__version__)
# exit()
def main():
    num_iter = 10000
    eta = 0.003
    num_hidden_neurons = [500, 125]
    activations = [tf.nn.tanh, tf.nn.sigmoid]
    nx = 10
    nt = 10

    # forward_euler(animate=False)
    diff_nn = tf_nn(nx, nt, num_hidden_neurons, activations, num_iter, eta)

main()
