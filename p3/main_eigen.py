from eigen import *

def main_tf(dx):
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

def main_fe(dt=0.0001):
    FE_eigen()


if __name__ == '__main__':
    main_fe()
