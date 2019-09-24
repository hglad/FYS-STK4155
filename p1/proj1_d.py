from proj1_funcs import *

np.random.seed(50)
# Make data
n = 50
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)

noise = np.asarray([np.random.normal(0,0.5,n*n)])
noise = np.reshape(noise, (n,n))
z = FrankeFunction(x,y) + noise

x_train, x_test, y_train, y_test, z_train, z_test = mselect.train_test_split( x,y,z , test_size=0.2, shuffle=True )
# z_true_train = np.ravel(FrankeFunction(x_test,y_test))
# z_true_test = np.ravel(FrankeFunction(x_test,y_test))

max_p = 15
polys = range(max_p)
l_ = np.logspace(np.log10(1e-5),np.log10(1),50)
# l_ = np.linspace(0.01,0.,max_p)
# l_ = [0]
print (l_)

R2_scores = np.zeros((max_p, len(l_)))
MSE_test = np.zeros((max_p, len(l_)))
MSE_train = np.zeros((max_p, len(l_)))
MSE_best_cv = np.zeros((max_p, len(l_)))

error_test = np.zeros((max_p, len(l_)))
bias_test = np.zeros((max_p, len(l_)))
var_test = np.zeros((max_p, len(l_)))

error_train = np.zeros((max_p, len(l_)))
bias_train = np.zeros((max_p, len(l_)))
var_train = np.zeros((max_p, len(l_)))
z_test = np.ravel(z_test)

# Perform cross-validation for different polynomial degrees

for i in range(len(l_)):
    for j in range(max_p):
        R2_scores[j,i], MSE_test[j,i], MSE_train[j,i], error_test[j,i], \
        bias_test[j,i], var_test[j,i], error_train[j,i], \
        bias_train[j,i], var_train[j,i] = cross_validation(x, y, z, k=5, p=j, l=l_[i], method='ridge')

# Find lambda and polynomial that gives best MSE
min_mse_coords = np.argwhere(MSE_test==MSE_test.min())
print (min_mse_coords)

best_poly = polys[ min_mse_coords[0,0] ]
best_lmbd = l_[ min_mse_coords[0,1] ]
print ("Polynomial degree with best R2:", best_poly)
print ("Lamdba with best R2:", best_lmbd)

plt.plot(polys, bias_test, '-b', label='bias (test)')
plt.plot(polys, var_test, '-r', label='variance (test)')
plt.plot(polys, MSE_test, '-g', label='MSE (test)')

plt.plot(polys, bias_train, '--b', label='bias (train)')
plt.plot(polys, var_train, '--r', label='variance (train)')
plt.plot(polys, MSE_train, '--g', label='MSE (train)')

# plt.axis([0,max_p,0,1])
plt.title('Prediction on train and test data')
plt.legend()
plt.show()

# Plot quantities as function of polynomial degree
plt.plot(polys, R2_scores, '-b',  label='R2_cv_test')
plt.plot(polys, MSE_test,  '-r',  label='MSE_cv_test')
plt.plot(polys, MSE_train, '--r', label='MSE_cv_train')
plt.legend()
plt.show()

z_true = FrankeFunction(x,y)
z_, z_pred  = predict_poly_ridge(x, y, z, best_poly, best_lmbd)

# Plot true data
plot_surf(x,y,z_true, color=cm.viridis, alpha=0.5)
# plot_surf(x,y,z_true, color=cm.Spectral)

# Plot predicted points
plot_points(x,y,z_pred)
plt.show()

print ("MSE (vs. true values):", metrics.mean_squared_error(np.ravel(z_true), z_pred))

# Plot MSE as function of lambda and polynomial degree
lmbd_mesh, poly_mesh = np.meshgrid(l_, polys)
plot_surf(poly_mesh, lmbd_mesh, MSE_test, color=cm.coolwarm, alpha=1)
plt.show()


#
