import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

tf.keras.backend.set_floatx('float64')  # Set default float type
tf.random.set_seed(42)

class DNModel(tf.keras.Model):
    def __init__(self):
        super(DNModel, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(
            100, activation=tf.keras.activations.sigmoid)
        self.dense_2 = tf.keras.layers.Dense(
            50, activation=tf.keras.activations.sigmoid)
        # self.dense_3 = tf.keras.layers.Dense(
        #     20, activation=tf.keras.activations.sigmoid)
        # self.dense_4 = tf.keras.layers.Dense(12, activation=tf.nn.sigmoid)

        self.out = tf.keras.layers.Dense(3, name='output')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        # x = self.dense_3(x)

        return self.out(x)


@tf.function
def analytical(x, t):
    return tf.sin(np.pi * x) * tf.exp(-np.pi * np.pi * t)


@tf.function
def I(x):
    return tf.sin(np.pi * x)


@tf.function
def trial_solution(model, x, t):
    point = tf.concat([x, t], axis=1)
    g = (1 - t) * I(x) + x * (1 - x) * t * model(point)

    return g

@tf.function
def trial_solution_eig(model, x0, t):
    g = tf.exp(-t)*x0 + (1 - t)*tf.exp(-t)*model(t)
    # g = tf.exp(-t)*x0*model(t)
    g = tf.transpose(g)

    return g


# Loss function
@tf.function
def loss(DNN, x0, t, A):
    # LHS: g'(t)
    # RHS: g(t).T*g(t)*A - g(t).T*A*g(t)
    with tf.GradientTape() as tape:
        tape.watch([x0, t])

        G = trial_solution_eig(DNN, x0, t)
        dG_dt = tape.gradient(G, t)
        print ("-----")
        print (G.get_shape())
        print ((tf.matmul(tf.transpose(G), G)).get_shape())
        print (A.get_shape())
        # exit()
        # term = tf.matmul(tf.matmul(tf.matmul(tf.transpose(G), G), A), G) \
        #  - tf.matmul(tf.matmul(tf.matmul(tf.transpose(G), A), G), G)
        term = tf.matmul( tf.matmul(tf.transpose(G), G)*A, G) \
         - tf.matmul(tf.matmul(tf.transpose(G), A), G)*G
        print ("term:", term.get_shape())

    return tf.losses.MSE(tf.zeros_like(dG_dt), dG_dt - term)


@tf.function
def grad(model, x, t, A):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, t, A)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def FE_eigen(dt=0.001):
    import sys
    np.random.seed(12)
    # Initialize arrays
    x = np.asarray([0, 0, 1])
    A = np.asarray([[3, 2, 4], [2, 0, 2], [4, 2, 3]])

    # Random symmetric matrix:
    # Q = np.random.randn(4,4)
    # A = (Q.T + Q)/2.

    # A = -A
    lmbd = np.matmul(np.matmul(x.T, A), x)/np.matmul(x.T, x)

    # Solution using numpy
    lmbd_true, v_true = np.linalg.eig(A)
    print (A, '\n')
    print (v_true, '\n')
    print (lmbd_true, '\n')
    eigv = np.max(lmbd_true)

    iters = 0
    max_iters = 100000
    eps = 1e-5

    while abs(eigv - lmbd) > eps and iters < max_iters:
        x = x + dt*(np.matmul( np.matmul(x.T, x)*A, x) \
        - np.matmul(np.matmul(x.T, A), x)*x)

        lmbd = np.matmul(np.matmul(x.T, A), x)/np.matmul(x.T, x)

        sys.stdout.write('lambda = %1.8f  \r' % (lmbd) )
        sys.stdout.flush()
        iters += 1

    print ("Computed eigenvector and eigenvalue:")
    print(x)
    print(lmbd)
    print (iters, "iters")


#
