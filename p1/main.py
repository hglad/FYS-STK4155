from functions import *

plt.rcParams.update({'font.size': 14})

"""
The main file does the following tasks:
    Perform OLS with data generated by the Franke function, without resampling
Then find the confidence interval of beta for p = 5.
"""
def main():
    n = 50
    dataset = 'Franke'
    method = sys.argv[3]

    if (method not in ['ols', 'ridge', 'lasso']):
        print ('Method \"%s\" not recognized. Please use one of the following methods: ols, ridge, lasso' % method)
        sys.exit()

    # np.random.seed(6)
    np.random.seed(10)
    print ("Dataset:", dataset)
    print ("Method:", method)

    # Determine dataset to analyze
    if (dataset == 'Franke'):
        x, y, z = Franke_dataset(n, noise=0.5)
        z_full = z
        nx = n
        ny = n
    else:
        z_full = DataImport('Norway_1arc.tif', sc=20)
        z_full = z_full/np.max(z_full)
        z = z_full/np.max(z_full)
        nx = len(z[0,:])
        ny = len(z[:,0])
        x = np.linspace(0,1,nx)
        y = np.linspace(0,1,ny)
        x, y = np.meshgrid(x, y)

    z_1d = np.ravel(z)

    # Define ranges for complexity and parameters
    min_p = int(sys.argv[1]);  max_p = int(sys.argv[2])
    polys = np.arange(min_p,max_p)

    min_param = 1e-6; max_param = 1e-1
    n_params = 10

    if (method == 'ols'):
        n_params = 1   # only run once for OLS, parameter value does not matter

    spacing = 'log'
    if (spacing == 'regular'):
        params = np.linspace(min_param, max_param, n_params)
    if (spacing == 'log'):
        params = np.logspace(np.log10(min_param), np.log10(max_param), n_params)

    # Initialize arrays
    R2_scores,MSE_test,MSE_train,MSE_best_cv,error_test,bias_test,var_test,\
    error_train = (np.zeros((len(polys), n_params)) for i in range(8))
    print ("Polynomials:", polys)
    print ("Parameters:", params)

    # Perform cross-validation with given set of polynomials and parameters
    for i in range(len(params)):
        for j in range(len(polys)):
            sys.stdout.write('poly %d/%d, param %d/%d  \r' % (j+1,len(polys),i+1,len(params)))
            sys.stdout.flush()

            R2_scores[j,i], MSE_test[j,i], MSE_train[j,i], error_test[j,i],  bias_test[j,i], var_test[j,i], error_train[j,i] = cross_validation(x, y, z, k=5, p=polys[j], dataset=dataset, param=params[i], method=method)

    print ("\n\n-----CV done-----")

    """
    Find best polynomial degree and hyperparameter using best MSE score. Code is
    different for OLS because we do not have a hyperparameter in that case.
    """
    if (method == 'ols'):
        poly_ind = np.argmin(MSE_test[:,0])
        best_poly = polys[poly_ind]
        param_ind = 0

    else:
        min_mse_coords = np.argwhere(MSE_test==MSE_test.min())
        poly_ind = min_mse_coords[0,0]
        param_ind = min_mse_coords[0,1]

    best_poly = polys[ poly_ind ]
    best_param = params[ param_ind ]
    print ("Polynomial degree with best MSE:", best_poly)
    print ("Param with best MSE: %1.3e" % best_param)
    print ("MSE:", MSE_test[poly_ind, param_ind])
    print("R2:", R2_scores[poly_ind, param_ind])

    # Visualize results
    # plot_bias_var_err(polys, bias_test, var_test, error_test, error_train)
    # plot_mse_train_test(polys, MSE_test, MSE_train, params, nx, ny)
    # plot_mse_poly_param(params, polys, MSE_test)
    plot_heatmap(params, polys, MSE_test)

    # Compare with true data if using Franke dataset
    if (dataset == 'Franke'):
        z = FrankeFunction(x,y)

    # visualize model (or dataset) as a 2D heatmap
    terrain_2d(x,y,z)

    # visualize model in 3D given colormap and transparency
    plot_surf(x,y,z,color=cm.coolwarm, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    UnitTest()
    main()
