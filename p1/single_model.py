from functions import *
"""
Perform a single prediction using predict_poly on given dataset. Data is then
visualized.
"""

dataset = 'Franke'

# Determine dataset to analyze
if (dataset == 'Franke'):
    n = 50
    x, y, z, z_true = Franke_dataset(n, noise=0.0)
    z_full = z
    nx = n
    ny = n
else:
    """
    Get real terrain data and reshape it. Normalize so that values are between
    0 and 1 for x,y,z.
    """
    z_full = DataImport('Norway_1arc.tif', sc=20)
    z_full = z_full/np.max(z_full)
    z = z_full/np.max(z_full)
    nx = len(z[0,:])
    ny = len(z[:,0])
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    x, y = np.meshgrid(x, y)

# Plot dataset
# plot_terrain_2d(z)
# plt.show()

z_, z_pred = predict_poly(x,y,z, 7, param=0, method='ols')

# PLot prediction
z_pred = np.reshape(z_pred, (ny,nx))
plot_surf(x,y,z,color=cm.coolwarm, alpha=1)
plt.show()
