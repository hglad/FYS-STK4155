import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import jacobian, hessian, grad
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
npr.seed(1)
np.random.seed(1)

def u(x):
    return np.sin(np.pi*x)

def u_e(point):
    x, t = point[0]
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)

class NeuralNetPDE:
    def __init__(self, x, t, config=[64,16], hidden_a_func=['tanh', 'sigmoid'], output_a_func='sigmoid'):
        self.config = config
        self.hidden_a_func = hidden_a_func
        self.output_a_func = output_a_func
        self.n_h_layers = len(config)
        self.nt = len(t)
        self.nx = len(x)
        self.x = x
        self.t = t
        # if len(x.shape) > 1:
        #     self.n_points = len(x.shape[0])
        # else:
        #     self.n_points = len(x)

        # print (x)
        # x = x.reshape(self.n_points, 1)    # column vector
        # print (x)

        # Set up arrays for NN structure
        self.a = np.empty(self.n_h_layers+2, dtype=np.ndarray)
        self.z = np.empty(self.n_h_layers+1, dtype=np.ndarray)
        self.w = np.empty(self.n_h_layers+1, dtype=np.ndarray)
        self.P = np.empty(self.n_h_layers+1, dtype=np.ndarray)

        # self.P[0] = npr.randn(2 + 1, config[0])     # (x, t) + bias
        self.P[0] = npr.randn(2, config[0])     # (x, t)

        for l in range(1, self.n_h_layers):
            self.P[l] = npr.randn(config[l-1], config[l])

        self.P[-1] = npr.randn(config[-1], 1)

    def activation(self, x, func):
        if (func == 'sigmoid'):
            t = 1./(1 + np.exp(-x))

        elif (func == 'softmax'):
            if len(x.shape) > 1:
                exp_term = np.exp(x)
                t = exp_term / np.sum(exp_term, axis=1, keepdims=True)
            else:
                exp_term = np.exp(x)
                t = exp_term / np.sum(exp_term)

        elif (func == 'tanh'):
            t = np.tanh(x)

        elif (func == 'relu'):
            inds = np.where(x < 0)
            x[inds] = 0
            t = x

        elif (func == ''):
            return x

        return t

    def N(self, point, P, num_coords = 2):
        num_points = point.shape[0]
        bias = np.ones((num_points, 1))
        # print (point.shape)
        # print (bias.shape)
        self.a[0] = point

        for l in range(1, self.n_h_layers+1):
            # print (l)
            # print (self.a[l-1].shape, self.P[l-1].shape)

            # self.a[l-1] = np.concatenate( (bias, self.a[l-1]), axis=1)
            self.z[l] = np.matmul(self.a[l-1], P[l-1]) + np.random.randn(num_points, 1)
            # self.a[l] = self.sigmoid(self.z[l])
            self.a[l] = self.activation(self.z[l], self.hidden_a_func[l-1])

        self.z[-1] = np.matmul(self.a[-2], P[-1]) + bias
        self.a_output = self.activation(self.z[-1], self.output_a_func)
        self.a[-1] = self.a_output

        return self.a_output[0][0]


    def trial_func(self, point, P):
        x, t = point[0]
        trial_val = (1-t)*u(x) + x*(1-x)*t*self.N(point, P)
        return trial_val
        # nx = 10
        # nt = 10
        # trial_vals = np.zeros(nx*nt)
        #
        # for j in range(nt):
        #     for i in range(nx):
        #         point = [x[i], t[j]]
        #         points[j*nx + i] = point
        #         trial_vals[j*nx + i] = (1-t[j])*u(x[i]) + x[i]*(1-x[i])*t[j]*self.N(point)

    def cost_function(self, P, x, t):
        total_cost = 0

        jac_func = jacobian(self.trial_func)
        hessian_func = hessian(self.trial_func)

        for x_ in x:
            for t_ in t:

                point = np.array([x_, t_])
                point = point.reshape(1,2)

                g_t = self.trial_func(point, P)
                g_t_jacobian = jac_func(point, P)
                g_t_hessian = hessian_func(point, P)

                # print (g_t_jacobian)
                # print (g_t_hessian)
                # exit()
                g_t_dt = g_t_jacobian[0][1]
                g_t_d2x = g_t_hessian[0][0][0][0]

                # print (g_t_dt)
                # print (g_t_d2x)
                # exit()

                cost = ( (g_t_dt - g_t_d2x))**2
                total_cost += cost

        print (total_cost)
        return total_cost

    def train(self, num_iter=10, lmbd=0.001):
        cost_function_grad = grad(self.cost_function, 1)
        P = self.P
        x = self.x
        t = self.t
        print (cost_function_grad)
        for k in range(num_iter):
            cost_grad = cost_function_grad(P, x, t)

            for l in range(self.n_h_layers+1):
                P[l] = P[l] - lmbd * cost_grad[l]

        self.P = P
        print('Final cost: ', self.cost_function(self.P, x, t))

    def results(self):

        ## Store the results
        g_dnn_ag = np.zeros((self.nx, self.nt))
        G_analytical = np.zeros((self.nx, self.nt))
        for i,x_ in enumerate(self.x):
            for j, t_ in enumerate(self.t):
                point = np.array([x_, t_])
                point = point.reshape(1,2)
                g_dnn_ag[i,j] = self.trial_func(point,self.P)

                G_analytical[i,j] = u_e(point)

        # Find the map difference between the analytical and the computed solution
        diff_ag = np.abs(g_dnn_ag - G_analytical)
        print('Max absolute difference between the analytical solution and the network: %g'%np.max(diff_ag))

        ## Plot the solutions in two dimensions, that being in position and time

        T,X = np.meshgrid(t,x)

        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')
        ax.set_title('Solution from the deep neural network w/ %d layer'%len(self.config))
        s = ax.plot_surface(T,X,g_dnn_ag,linewidth=0,antialiased=False,cmap=cm.viridis)
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('Position $x$');


        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')
        ax.set_title('Analytical solution')
        s = ax.plot_surface(T,X,G_analytical,linewidth=0,antialiased=False,cmap=cm.viridis)
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('Position $x$');

        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')
        ax.set_title('Difference')
        s = ax.plot_surface(T,X,diff_ag,linewidth=0,antialiased=False,cmap=cm.viridis)
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('Position $x$');

        ## Take some slices of the 3D plots just to see the solutions at particular times
        indx1 = 0
        indx2 = int(self.nt/2)
        indx3 = self.nt-1

        t1 = self.t[indx1]
        t2 = self.t[indx2]
        t3 = self.t[indx3]

        # Slice the results from the DNN
        res1 = g_dnn_ag[:,indx1]
        res2 = g_dnn_ag[:,indx2]
        res3 = g_dnn_ag[:,indx3]

        # Slice the analytical results
        res_analytical1 = G_analytical[:,indx1]
        res_analytical2 = G_analytical[:,indx2]
        res_analytical3 = G_analytical[:,indx3]

        # Plot the slices
        plt.figure(figsize=(10,10))
        plt.title("Computed solutions at time = %g"%t1)
        plt.plot(self.x, res1)
        plt.plot(self.x, res_analytical1)
        plt.legend(['dnn','analytical'])

        plt.figure(figsize=(10,10))
        plt.title("Computed solutions at time = %g"%t2)
        plt.plot(self.x, res2)
        plt.plot(self.x, res_analytical2)
        plt.legend(['dnn','analytical'])

        plt.figure(figsize=(10,10))
        plt.title("Computed solutions at time = %g"%t3)
        plt.plot(self.x, res3)
        plt.plot(self.x, res_analytical3)
        plt.legend(['dnn','analytical'])

        plt.show()

nx = 10
nt = 10
x = np.linspace(0,1,nx)
t = np.linspace(0,1,nt)

nn = NeuralNetPDE(x, t, config=[100], hidden_a_func=['sigmoid'], output_a_func='sigmoid')
nn.train(lmbd=0.0001, num_iter=10)
nn.results()


# points = np.empty(len(x)*len(t), dtype=np.ndarray)
#
# for i in range(nt):
#     for j in range(nx):
#         points[i*nx + j] = np.array([ x[j], t[i] ])

# points = np.zeros((nx*nt, 2))
# for j in range(nt):
#     for i in range(nx):
#         # points[j*nx + i] = [x[i], t[j]]
#         points[j*nx + i] = np.array(x[i], t[j])




"""
    def cost_function(self, points):
        total_cost = 0

        jac_func = jacobian(self.trial_func)
        hessian_func = hessian(self.trial_func)

        for k in range(self.nx*self.nt):

            point = points[k].reshape(1,2)

            g_t = self.trial_func(point)
            g_t_jacobian = jac_func(point)
            g_t_hessian = hessian_func(point)
            # print (g_t_jacobian)
            # print (g_t_hessian)
            # exit()
            g_t_dt = g_t_jacobian[0][0]
            g_t_d2x = g_t_hessian[0][0][0][1]

            # print (g_t_dt)
            # print (g_t_d2x)
            # exit()

            cost = ( (g_t_dt - g_t_d2x))**2
            total_cost += cost

        print (total_cost)
        return total_cost
"""
