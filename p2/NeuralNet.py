import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.compose import ColumnTransformer

class NeuralNet:
    def __init__(self, X, y, neuron_lengths=[16,8], hidden_a_func=['sigmoid', 'tanh'], output_a_func='softmax', type='class'):
        self.X = X
        self.type = type

        if (type == 'class'):
            self.n_categories = np.max(y+1)
            if output_a_func == 'sigmoid':
                self.n_categories = 1

        if (type == 'reg'):
            self.n_categories = 1

        self.hidden_a_func = hidden_a_func
        self.output_a_func = output_a_func

        if self.n_categories > 1:
            onehotencoder = OneHotEncoder(categories="auto", sparse=False)

            y = ColumnTransformer(
                [("", onehotencoder, [0]),],
                remainder="passthrough"
            ).fit_transform(y)

        self.y = y
        self.iters_done = 0

        if len(X.shape) > 1:
            self.n_train, self.m_train  = X.shape

        # Special case for a single training point
        else:
            self.X = np.reshape(X, (1, len(X)))
            self.y = np.reshape(y, (1,1))
            self.n_train = 1
            self.m_train = len(X)

        # Used for grid search function
        self.best_accuracy = -1
        self.best_config = []
        self.best_gamma = 0
        self.best_lmbd = 0

        # create structure of activations, weight and bias arrays
        self.create_structure(neuron_lengths)


    def print_properties(self):
        print ("----Neural network----")
        print (self.m_train, "input values")
        print (self.n_h_layers, "hidden layers")
        print (self.n_h_neurons, "neurons per hidden layer")
        print (self.n_categories, "output categories\n")


    def create_structure(self, neuron_lengths):
        np.random.seed(1)
        # Do not include layers that have 0 neurons
        self.n_h_neurons = []
        for layer in neuron_lengths:
            if (layer != 0):
                self.n_h_neurons.append(layer)

        self.n_h_layers = len(self.n_h_neurons)

        # Check if hidden layers have corresponding activation functions
        if self.n_h_layers != len(self.hidden_a_func):
            print("Num. of hidden layers does not match num. of hidden activation functions.")
            exit()

        self.a = np.empty(self.n_h_layers+2, dtype=np.ndarray)
        self.z = np.empty(self.n_h_layers+1, dtype=np.ndarray)
        self.w = np.empty(self.n_h_layers+1, dtype=np.ndarray)
        self.b = np.empty(self.n_h_layers+1, dtype=np.ndarray)

        self.a[0] = self.X

        # Input layer -> first hidden layer weights
        self.w[0] = np.random.uniform(-1, 1, (self.m_train, self.n_h_neurons[0]))

        # Hidden layers
        for l in range(self.n_h_layers):
            self.b[l] = np.random.uniform(-0.01, 0.01, (self.n_h_neurons[l]))
            self.a[l+1] = np.zeros(self.n_h_neurons[l])
            self.z[l+1] = np.zeros(self.n_h_neurons[l])

        # Hidden layers (weights)
        for l in range(1, self.n_h_layers):
            self.w[l] = np.random.uniform(-1, 1, (self.n_h_neurons[l-1], self.n_h_neurons[l]))

        self.b[-1] = np.random.uniform(-0.01, 0.01,(self.n_categories))
        self.w[-1] = np.random.uniform(-1, 1, (self.n_h_neurons[-1], self.n_categories))
        self.a[-1] = np.zeros(self.n_categories)  # Output layer
        self.z[-1] = np.zeros(self.n_categories)


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


    def activation_der(self, x, func):
        if func == 'sigmoid':
            return x*(1 - x)

        elif func == 'tanh':
            return 1 - x**2

        elif func == 'relu':
            inds1 = np.where(x > 0)
            inds2 = np.where(x <= 0)
            x[inds1] = 1
            x[inds2] = 0

            return x

        elif func == 'softmax':
            return 1

        elif func == '':
            return 1


    def feed_forward(self):
        """
        activation in hidden layer: take sigmoid of weighted input,
        a_l = sigmoid( (weights*previous activation) + bias )
        first activation in NN is the input data
        """
        # Iterate through hidden layers
        for l in range(1, self.n_h_layers+1):
            self.z[l] = np.matmul(self.a[l-1], self.w[l-1]) + self.b[l-1]
            self.a[l] = self.activation(self.z[l], self.hidden_a_func[l-1])

        # Output layer
        self.z[-1] = np.matmul(self.a[-2], self.w[-1]) + self.b[-1]

        self.a_output = self.activation(self.z[-1], self.output_a_func)
        self.a[-1] = self.a_output


    def w_b_gradients(self, delta, l):
        b_grad = np.sum(delta, axis=0)
        w_grad = np.matmul(self.a[l].T, delta)
        if self.lmbd > 0.0:
            w_grad += self.lmbd * self.w[l]

        return w_grad, b_grad


    def back_propagation(self):
        if self.type == 'class':
            delta_L = self.activation_der(self.a[-1], self.output_a_func)*(self.a_output - self.y) # error in output layer
        if self.type == 'reg':
            delta_L = self.a_output - self.y

        # Output layer
        w_grad, b_grad = self.w_b_gradients(delta_L, self.n_h_layers)
        self.w[-1] -= self.gamma * w_grad
        self.b[-1] -= self.gamma * b_grad

        delta_old = delta_L

        # Loop from last hidden layer to first hidden layer
        for l in range(self.n_h_layers, 0, -1):
            # Use previous error to propagate error back to first hidden layer
            delta_h = np.matmul(delta_old, self.w[l].T) * self.activation_der(self.a[l], self.hidden_a_func[l-1])
            # delta_h = np.matmul(self.w[l].T, delta_old) *
            w_grad, b_grad = self.w_b_gradients(delta_h, l-1)

            # Optimize weights/biases
            self.w[l-1] -= self.gamma * w_grad
            self.b[l-1] -= self.gamma * b_grad
            delta_old = delta_h     # Update previous error

        self.iters_done += 1
        # sys.stdout.write('iter %d / %d , loss %1.3e \r' % (self.iters_done, self.iters, total_loss))
        # sys.stdout.flush()


    def train(self, iters=2000, gamma=1e-3, lmbd=0):
        self.print_properties()
        """
        Perform feed-forward and back propagation for given number of iterations
        """
        self.gamma = gamma
        self.lmbd = lmbd
        self.iters = iters

        for i in range(iters):
            self.feed_forward()
            self.back_propagation()


    def predict(self, X_test):
        self.a[0] = X_test
        self.feed_forward()

        if len(X_test.shape) > 1:
            n, m = X_test.shape
            y_pred = np.argmax(self.a_output, axis=1)
        else:
            n = len(X_test)
            X_test = np.reshape(X_test, (n,1))
            y_pred = np.argmax(self.a_output)

        return y_pred


    def predict_single_output_neuron(self, X_test):
        self.a[0] = X_test
        self.feed_forward()
        y_pred = np.zeros(X_test.shape[0])
        # For loop because vectorized version behaved strangely
        for i in range(len(self.a_output)):
            y_pred[i] = np.round(self.a_output[i])

        return y_pred


    def predict_regression(self, X_test):
        self.a[0] = X_test
        self.feed_forward()
        y_pred = self.a_output

        return y_pred


    def grid_search(self, X_test, y_test, params, gammas, *config):
        """
        Function used for determining best combination of learning rate, penalty and
        neuron configuration.
        """
        config = config[0]
        heatmap_array = np.zeros((len(params), len(gammas)))
        for i in range(len(params)):
            for j in range(len(gammas)):
                self.create_structure(config)
                self.train(iters=3000, gamma=gammas[j], lmbd=params[i])

                if self.n_categories == 1:
                    y_pred = self.predict_single_output_neuron(X_test)
                else:
                    y_pred = self.predict(X_test)

                print ("gamma =", gammas[j])
                print ("lmbd = ", params[i])
                accuracy = np.mean(y_pred == y_test[:,0])
                heatmap_array[i, j] = accuracy

                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_config = config
                    self.best_lmbd = params[i]
                    self.best_gamma = gammas[j]

                print ("accuracy =", accuracy)
                print ("best =", self.best_accuracy, "with", self.best_config, "lmbd =", self.best_lmbd, "gamma =", self.best_gamma)
                print ("--------------\n")

        xtick = gammas
        ytick = params
        sb.heatmap(heatmap_array, annot=True, fmt="g", xticklabels=xtick, yticklabels=ytick)
        plt.xlabel('learning rate $\gamma$')
        plt.ylabel('penalty $\lambda$')
        plt.show()

    def return_params(self):
        return self.best_accuracy, self.best_config, self.best_lmbd, self.best_gamma
