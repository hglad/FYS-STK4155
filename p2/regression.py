from NeuralNet import *
from functions import *
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def main_NN():
    n = 50
    dataset = 'Franke'      # set to any other string to use terrain data

    np.random.seed(1)
    print ("Dataset:", dataset)

    # Determine dataset to analyze
    if (dataset == 'Franke'):
        X, x_grid, y_grid, z = Franke_dataset(n, noise=0.1)
        z = np.reshape(z, (n*n, 1))
    else:
        z_full = DataImport('Norway_1arc.tif', sc=20)
        z_full = z_full/np.max(z_full)
        z = z_full/np.max(z_full)
        nx = len(z[0,:])
        ny = len(z[:,0])
        x = np.linspace(0,1,nx)
        y = np.linspace(0,1,ny)
        x_grid, y_grid = np.meshgrid(x, y)


    iters = 5000
    lmbd = 0.0; gamma = 1e-5

    n_categories = 1
    hidden_a_func = ['tanh','tanh']
    output_a_func = ''

    n_params = 5
    n_gammas = 4
    params = np.zeros(n_params)
    params[1:] = np.logspace(1, -2, n_params-1)
    gammas = np.logspace(-2, -5, n_gammas)

    print(params)
    print(gammas)

    # test_oneside = 0.25
    # n_test = int(n*test_oneside)
    # test_size = test_oneside**2 - test_oneside/n**2
    test_frac = 0.25
    n_test = int(test_frac*n**2)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_frac, random_state=123)

    train_single_NN = True
    # plot_surf(x_grid, y_grid, z.reshape(n,n), cm.coolwarm, 1)
    # plt.show()

    if train_single_NN == True:
        # config = [4,16,4]
        config = [80,80]
        # config = [16,8,4]
        NN = NeuralNet(X_train, z_train, config, hidden_a_func, output_a_func, 'reg')
        NN.train(iters, gamma, lmbd=lmbd)
        z_pred = NN.predict_regression(X_test)

        r2_score = metrics.r2_score(z_test, z_pred)
        mse = metrics.mean_squared_error(z_test, z_pred)


        print ("gamma =", gamma)
        print ("lmbd =", lmbd)
        print ("r2 =", r2_score)
        print ("mse =", mse)

        print ("--------------\n")


        # x_test, y_test = np.meshgrid(X_test[:,0], X_test[:,1])
        # n_test_1d = int(np.sqrt(n_test))
        # x_grid, y_grid = get_grid(X_test)

        # plot_surf(x_grid, y_grid, z_pred.reshape(n_test_1d, n_test_1d), cm.coolwarm)
        # plot_surf(x_grid, y_grid, z_test.reshape(n_test_1d, n_test_1d), cm.gray, alpha=0.5)
        # plt.show()

        # plt.imshow(z_pred.reshape(n,n))
        # plt.show()
        #
        # plt.imshow(z_train.reshape(n,n))
        # plt.show()

    exit()

    NN_grid = NeuralNet(X_train, y_train, [32, 16], ['tanh', 'sigmoid'], 'softmax')
    NN_grid.grid_search(X_test, y_test, params, gammas, [32, 16])

    # Iterate over multiple hidden layer configurations
    # for i in range(1, 6):
    #     for j in range(0, i+1):
    #         for k in range(0, 1):
    #             config = [i, j, k]
    #             NN_grid.grid_search(X_test, y_test, params, gammas, config)

    best_accuracy, best_config,best_lmbd, best_gamma = NN_grid.return_params()

    print ("\n--- Grid search done ---")
    print ('Best accuracy:', best_accuracy)
    print ("with configuration", best_config, "lmbd =", best_lmbd, "gamma =", best_gamma)


if __name__ == '__main__':
    main_NN()
