import os
os.environ['KMP_WARNINGS'] = 'off'
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import jacobian, hessian, grad
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import tensorflow.compat.v1 as tf
# import tensorflow as tf
from tensorflow import keras
tf.logging.set_verbosity(tf.logging.ERROR)

# tf.disable_v2_behavior()
def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass
tensorflow_shutup()

def u(x):
    return tf.sin(np.pi*x)

def u_e(x, t):
    return tf.exp(-np.pi**2*t)*tf.sin(np.pi*x)

def u_e_np(x, t):
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)


def tf_nn(nx, nt, num_hidden_neurons, num_iter=100000, eta=0.01):
    tf.reset_default_graph()

    # Set a seed to ensure getting the same results from every run
    tf.set_random_seed(4155)
    nx = 10
    nt = 10

    x_np = np.linspace(0,1,nx)
    t_np = np.linspace(0,1,nt)

    X,T = np.meshgrid(x_np, t_np)

    x = X.ravel()
    t = T.ravel()

    ## The construction phase
    zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)),shape=(-1,1))
    x = tf.reshape(tf.convert_to_tensor(x),shape=(-1,1))
    t = tf.reshape(tf.convert_to_tensor(t),shape=(-1,1))

    pts = tf.concat([x,t],1)       # input layer
    num_hidden_layers = len(num_hidden_neurons)

    X = tf.convert_to_tensor(X)
    T = tf.convert_to_tensor(T)

    # Define layer structure
    with tf.name_scope('dnn'):
        num_hidden_layers = np.size(num_hidden_neurons)
        activations = [tf.nn.sigmoid, tf.nn.sigmoid]
        previous_layer = pts

        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l], name=('hidden%d' %(l+1)), activation=activations[l])

            previous_layer = current_layer

        dnn_output = tf.layers.dense(previous_layer, 1, name='output', activation=None)

    # Define loss function
    # trial function satisfies boundary conditions and initial condition
    with tf.name_scope('loss'):
        g_t = (1 - t)*u(x) + x*(1 - x)*t*dnn_output
        g_t_d2x = tf.gradients(tf.gradients(g_t, x), x)
        g_t_dt = tf.gradients(g_t, t)
        loss = tf.losses.mean_squared_error(zeros, g_t_dt[0] - g_t_d2x[0])

    # Define optimizer
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(eta)
        training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    g_e = u_e(x, t)
    # g_dnn = None

    with tf.Session() as sess:
        init.run()
        for i in range(num_iter):
            sess.run(training_op)

            if i % 1000 == 0:
                print (loss.eval())
                # g_e = g_e.eval()
                # g_dnn = g_t.eval()
            #
            #     plot_g_e = g_e.eval().reshape((nt, nx))
            #     plot_g_dnn = g_t.eval().reshape((nt, nx))
            #
            #     plt.plot(x_np, plot_g_e[int(nt/2), :])
            #     plt.plot(x_np, plot_g_dnn[int(nt/2), :])
            #     plt.axis([0,1,0,0.1])
            #     plt.pause(0.001)
            #     plt.clf()

        g_e = g_e.eval()        # analytical solution
        g_dnn = g_t.eval()      # NN solution


    diff = np.abs(g_e - g_dnn)
    print('Max absolute difference between analytical solution and TensorFlow DNN ',np.max(diff))

    G_e = g_e.reshape((nt,nx))
    G_dnn = g_dnn.reshape((nt,nx))
    diff = diff.reshape((nt, nx))

    # Plot the results
    X,T = np.meshgrid(x_np, t_np)

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.set_title('Solution from the deep neural network w/ %d layer'%len(num_hidden_neurons))
    s = ax.plot_surface(X,T,G_dnn,linewidth=0,antialiased=False,cmap=cm.viridis)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Position $x$');

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.set_title('Analytical solution')
    s = ax.plot_surface(X,T,G_e,linewidth=0,antialiased=False,cmap=cm.viridis)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Position $x$');

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.set_title('Difference')
    s = ax.plot_surface(X,T,diff,linewidth=0,antialiased=False,cmap=cm.viridis)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Position $x$');

    ## Take some 3D slices
    indx1 = 0
    indx2 = int(nt/2)
    indx3 = nt-1

    t1 = t_np[indx1]
    t2 = t_np[indx2]
    t3 = t_np[indx3]

    # Slice the results from the DNN
    res1 = G_dnn[indx1,:]
    res2 = G_dnn[indx2,:]
    res3 = G_dnn[indx3,:]

    # Slice the analytical results
    res_analytical1 = G_e[indx1,:]
    res_analytical2 = G_e[indx2,:]
    res_analytical3 = G_e[indx3,:]

    # Plot the slices
    plt.figure(figsize=(10,10))
    plt.title("Computed solutions at time = %g"%t1)
    plt.plot(x_np, res1)
    plt.plot(x_np, res_analytical1)
    plt.legend(['dnn','analytical'])

    plt.figure(figsize=(10,10))
    plt.title("Computed solutions at time = %g"%t2)
    plt.plot(x_np, res2)
    plt.plot(x_np, res_analytical2)
    plt.legend(['dnn','analytical'])

    plt.figure(figsize=(10,10))
    plt.title("Computed solutions at time = %g"%t3)
    plt.plot(x_np, res3)
    plt.plot(x_np, res_analytical3)
    plt.legend(['dnn','analytical'])

    plt.show()

    return diff

def forward_euler(animate=False):
    # Discretization parameters
    dx = 1./10.
    L = 1
    nx = int(L/dx)

    dt = 0.1*dx**2
    T = 1
    nt = int(T/dt)
    CX = dt/dx**2

    print (nx, nt)

    # Initialize arrays
    u = np.zeros((nt, nx))
    u_sc = np.zeros(nx)
    t = np.linspace(0, T, nt)
    x = np.linspace(0, L, nx)

    # Initial values
    u0 = np.sin(np.pi*x)
    u0[0] = u0[-1] = 0
    u[0] = u0

    # u1 = u0             # u1 = solution at previous time step
    # for j in range(nt):
        # u[1:-1] = u1[1:-1] + CX*(u1[2:] - 2*u1[1:-1] + u1[0:-2]) # vectorized
        # u1 = u

    for j in range(1,nt):
        u[j, 1:-1] = u[j-1, 1:-1] + CX*(u[j-1, 2:] - 2*u[j-1, 1:-1] + u[j-1, 0:-2]) # vectorized
        u1 = u

        if animate:
            plt.plot(x, u_e_np(x, t[j]))
            plt.plot(x, u[j], 'r--')
            plt.axis([0,1,0,1])
            plt.pause(1)
            plt.clf()

    # Compare to analytical solution
    x, t = np.meshgrid(x, t)
    diff = np.abs(u - u_e_np(x,t))
    print (np.max(diff))
