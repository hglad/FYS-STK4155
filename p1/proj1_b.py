from proj1_funcs import *
np.random.seed(0)
# Make data
n = 100
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)

noise = np.asarray([np.random.normal(0,0.1,n*n)])
noise = np.reshape(noise, (n,n))
z = FrankeFunction(x,y) + noise

x_train, x_test, y_train, y_test, z_train, z_test = mselect.train_test_split( x,y,z , test_size=0.2 )

max_p = 20
polys = range(1, max_p+1)
R2_scores = np.zeros(max_p)
MSE_values = np.zeros(max_p)
MSE_best_cv = np.zeros(max_p)

# Perform cross-validation for different polynomial degrees
j = 0
for p in polys:
    best_beta, MSE_best_cv[j] = cross_validation(x_train, y_train, z_train, 10, p=polys[j])
    
    # Predict using original test data (from train_test_split)
    X_test = CreateDesignMatrix_X(x_test, y_test, polys[j])
    z_pred = X_test @ best_beta
    z_test_1d = np.ravel(z_test)

    R2_scores[j] = metrics.r2_score(z_test_1d, z_pred)
    MSE_values[j] = metrics.mean_squared_error(z_test_1d, z_pred)
    j += 1

# Plot model scores
plt.plot(polys, R2_scores, '-b', label='R2_test')
plt.plot(polys, MSE_values, '-r', label='MSE_test')
plt.legend()
plt.show()
# Compare MSE for train and test data
plt.semilogy(polys, MSE_best_cv, '-b', label='MSE_train')
plt.semilogy(polys, MSE_values, '-r', label='MSE_test')
plt.legend()
plt.show()

# Regression
# X = CreateDesignMatrix_X(x_train, y_train, p)
# z_ = np.ravel(z_train)			# flattened array
# beta = np.linalg.inv( np.dot(X.T, X) ) .dot(X.T) .dot(z_)
#
# # Create prediction using training points
# z_pred = X @ beta




#
