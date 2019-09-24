from proj1_funcs import *

z = DataImport('Norway_1arc.tif', sc=20)

nx = len(z[0,:])
ny = len(z[:,0])
x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)
x, y = np.meshgrid(x, y)

# Normalize
z = z/(np.max(z))
# plt.imshow(z, cmap='gray')
# plt.show()
max_p = 12
num_lmbd = 100
polys = range(max_p)
l_ = np.logspace(np.log10(1e-4),np.log10(1e-3),num_lmbd)

R2_scores = np.zeros((max_p, len(l_)))
MSE_test = np.ones((max_p, len(l_)))
MSE_train = np.ones((max_p, len(l_)))
MSE_best_cv = np.zeros((max_p, len(l_)))

error_test = np.zeros((max_p, len(l_)))
bias_test = np.zeros((max_p, len(l_)))
var_test = np.zeros((max_p, len(l_)))

error_train = np.zeros((max_p, len(l_)))
bias_train = np.zeros((max_p, len(l_)))
var_train = np.zeros((max_p, len(l_)))

# for i in range(len(l_)):
#     for j in range(max_p):
#         R2_scores[j,i], MSE_test[j,i], MSE_train[j,i], error_test[j,i], \
#         bias_test[j,i], var_test[j,i], error_train[j,i], \
#         bias_train[j,i], var_train[j,i] = cross_validation(x, y, z, k=5, p=j, l=l_[i], method='ridge')

for i in range(1):
    for j in range(max_p):
        R2_scores[j,i], MSE_test[j,i], MSE_train[j,i], error_test[j,i], \
        bias_test[j,i], var_test[j,i], error_train[j,i], \
        bias_train[j,i], var_train[j,i] = cross_validation(x, y, z, k=5, p=j, method='ols')

min_mse_coords = np.argwhere(MSE_test==MSE_test.min())
print (min_mse_coords)

best_poly = polys[ min_mse_coords[0,0] ]
best_lmbd = l_[ min_mse_coords[0,1] ]
print ("Polynomial degree with best R2:", best_poly)
print ("Lamdba with best R2:", best_lmbd)

z_, z_pred  = predict_poly_ols(x, y, z, best_poly)
# dim = int(np.floor(np.sqrt(len(z_pred))))
z_pred = np.reshape(z_pred, (nx,ny))
# print (z_pred.shape)
plot_surf(x,y, z_pred, color=cm.viridis)
# plt.imshow(z_pred, cmap='gray')
plt.show()

#
