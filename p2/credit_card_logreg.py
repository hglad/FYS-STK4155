from functions import *

def main():
    dataset = int(sys.argv[1])
    X, y = load_dataset(dataset)


    # Split into train and test data
    print (X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    # logreg_sklearn(X_train, X_test, y_train, y_test)
    my_logreg(X_train, X_test, y_train, y_test)


def main_NN():
    from sklearn.neural_network import MLPClassifier
    dataset = int(sys.argv[1])
    X, y = load_dataset(dataset)
    n, m = X.shape
    # print (y.shape)

    iters = 10000
    gamma = 1e-6

    NN = NeuralNet(X, y, n_h_layers=2, n_h_neurons=int(m/5.), n_categories=2)
    NN.fit(iters, gamma)
    y_pred = NN.predict(y[:,0])
    ConfMatrix(y[:,0], y_pred)

    # scikit-learn NN
    scikit_NN = MLPClassifier(solver='lbfgs', alpha=0, learning_rate='constant', learning_rate_init=gamma, activation='logistic', hidden_layer_sizes=int(m), random_state=1,max_iter=iters)
    scikit_NN.fit(X, y[:,0])
    y_pred = scikit_NN.predict(X)
    ConfMatrix(y[:,0], y_pred)


if __name__ == '__main__':
    main_NN()

# dataset 1: 100 % accuracy, random_state=123, test_size=0.3
# iters = 50000
# gamma = 5e-2
# beta_0 = np.random.uniform(-10000,10000,m)         # random initial weights

# dataset 0:
# Accuracy: 0.824912 (gamma = 1e-06, 500000 iters)
# Correctly classified: 7053
# Default classified as non-default: 1175
# Non-default classified as default: 322






#
