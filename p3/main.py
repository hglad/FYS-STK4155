from tf_pde import *

def main():
    num_iter = 100000
    eta = 0.01
    num_hidden_neurons = [100, 20]
    nx = 10
    nt = 10

    # forward_euler(animate=False)
    diff_nn = tf_nn(nx, nt, num_hidden_neurons, num_iter, eta)

main()
