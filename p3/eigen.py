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


def main(dx):
    n = 3
    Nx = 10
    Nt = 31
    # A = tf.random.uniform([n, n], -1, 1, dtype=tf.float64, seed=42)
    A = tf.constant([[3, 2, 4], [2, 0, 2], [4, 2, 3]], dtype=tf.float64)

    start = tf.constant(0, dtype=tf.float64)
    stop = tf.constant(10, dtype=tf.float64)
    X0 = tf.constant([0.2, 0.5, 0.8], dtype=tf.float64)
    t = tf.linspace(start, stop, Nt)
    # X, T = np.meshgrid(tf.reshape(tf.linspace(start, stop, Nx), (-1, 1)),
                       # tf.reshape(tf.linspace(start, stop, Nt), (-1, 1)))

    x0, t = tf.reshape(X0, (-1, 1)), tf.reshape(t, (-1, 1))
    learning_rate = 0.01
    epochs = 100

    model = DNModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Training
    lmbd_true, v_true = np.linalg.eig(A)
    print (lmbd_true)
    print (v_true)
    # v = x0

    for t_ in t:
        t_ = tf.reshape(t_, (-1, 1))
        for epoch in range(epochs):
            cost, gradients = grad(model, x0, t_, A)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            print('  \r', end='')
            print(f'Step: {optimizer.iterations.numpy()}, '
                  + f'Loss: {tf.reduce_mean(cost.numpy())}', flush=True, end='')

        print('')

        u_dnn = trial_solution_eig(model, x0, t_)
        # u_dnn = tf.reshape(u_dnn, (Nt, n))

        # exit()
        v = u_dnn
        term1 = tf.matmul(tf.matmul(tf.transpose(v), A), v)
        term2 = tf.matmul(tf.transpose(v), v)

        lmbd = term1/term2
        tf.print(lmbd)

    # xx = np.linspace(0, 1, Nx)
    # tt = np.linspace(0, 10, Nt)


if __name__ == '__main__':
    main(0.1)
