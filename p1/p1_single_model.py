from proj1_funcs import *

dataset = 'Franke'

# Determine dataset to analyze
if (dataset == 'Franke'):
    n = 50
    x, y, z = Franke_dataset(n, noise=0.0)
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

# Plot dataset
plot_surf(x,y,z,color=cm.coolwarm, alpha=1)
plt.show()

z_, z_pred = predict_poly(x,y,z,6, param=1.274e-4, method='ridge')

# PLot prediction
z_pred = np.reshape(z_pred, (nx,ny))
plot_surf(x,y,z_pred,color=cm.coolwarm, alpha=1)
plt.show()
