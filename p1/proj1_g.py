from proj1_funcs import *
plt.rcParams.update({'font.size': 12})

z = DataImport('Norway_1arc.tif', sc=10)

nx = len(z[0,:])
ny = len(z[:,0])
x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)
x, y = np.meshgrid(x, y)
print (len(x), len(y))

# Normalize
z = z/(np.max(z))
# plt.show()
# plt.figure()
# plt.imshow(z, cmap='gray')
min_p = 5
max_p = 40
n_param = 1
n_polys = abs(max_p-min_p)
polys = range(min_p, max_p)
params = np.logspace(np.log10(1e-6),np.log10(1e-2),n_param)
print (params[0], params[-1])

R2_scores = np.zeros((n_polys, n_param))
MSE_test = np.ones((n_polys, n_param))
MSE_train = np.ones((n_polys, n_param))
MSE_best_cv = np.zeros((n_polys, n_param))

error_test = np.zeros((n_polys, n_param))
bias_test = np.zeros((n_polys, n_param))
var_test = np.zeros((n_polys, n_param))

error_train = np.zeros((n_polys, n_param))
bias_train = np.zeros((n_polys, n_param))
var_train = np.zeros((n_polys, n_param))

# Determine statistical quantities with CV for every polynomial degree
# i = 0
# for j in range(len(polys)):
#     R2_scores[j,i], MSE_test[j,i], MSE_train[j,i], error_test[j,i], \
#     bias_test[j,i], var_test[j,i], error_train[j,i], \
#     bias_train[j,i], var_train[j,i] = cross_validation(x, y, z, k=5, p=polys[j], method='ols')
#     print (j)

# Ridge or lasso
# print ('start')
# for i in range(len(params)):
#     for j in range(len(polys)):
#         R2_scores[j,i], MSE_test[j,i], MSE_train[j,i], error_test[j,i], \
#         bias_test[j,i], var_test[j,i], error_train[j,i], \
#         bias_train[j,i], var_train[j,i] = cross_validation(x, y, z, k=5, p=polys[j], param=params[i], method='ols')
#     print (i)

min_mse_coords = np.argwhere(MSE_test==MSE_test.min())
print (min_mse_coords)

best_poly = polys[ min_mse_coords[0,0] ]
best_param = params[ min_mse_coords[0,1] ]
print ("Polynomial degree with best R2:", best_poly)
print ("Param with best R2:", best_param)

p = 20
alpha = 2e-7
max_iter = 5000     # for lasso

# R2, MSE_test, MSE_train, error_test, bias_test, var_test, \
# error_train, bias_train, var_train = cross_validation(x,y,z, k=5, p=best_poly, l=best_param, alpha=alpha, method='ridge')
# print(R2, MSE_test, MSE_train)

z_, z_pred  = predict_poly_ols(x, y, z, 39)
# z_, z_pred = predict_poly_ridge(x,y,z, best_poly, best_param)
# z_, z_pred = predict_lasso(x,y,z, best_poly, best_param, max_iter=5000)

# print("p = %d, alpha = %f, max_iter = %d" % (p, alpha, max_iter))
print ("MSE:", metrics.mean_squared_error(np.ravel(z), z_pred))
print("R2:", metrics.r2_score(np.ravel(z), z_pred))
z_pred = np.reshape(z_pred, (ny, nx))

# print (z_pred.shape)
# plt.figure()
# plt.imshow(z_pred, cmap='gray')

plot_surf(x,y,z,color=cm.hot,alpha=0.25)
plot_points(x,y, z_pred)
plt.show()


"""
Polynomial degree with best MSE: 40
Param with best MSE: 2.782559402207126e-06       # lambda (ridge)
MSE: 0.009412294287341716
R2: 0.6647522947319482

Polynomial degree with best R2: 35
Param with best R2: 1e-06
MSE: 0.009248891577062974
R2: 0.670572383010446

Polynomial degree with best R2: 39
(ols)
MSE: 0.007917198510529312
R2: 0.7180047125836082
"""






#
