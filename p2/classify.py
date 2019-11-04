from functions import *
from NeuralNet import *

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

    iters = 5000
    lmbd = 0; gamma = 5e-6

    n_categories = 10

    n_params = 7
    n_gammas = 5
    params = np.zeros(n_params)
    params[1:] = np.logspace(1, -2, n_params-1)
    gammas = np.logspace(-4, -6, n_gammas)

    print(params)
    print(gammas)

    # neuron_lengths_h1 = np.arange(10, 33)
    # neuron_lengths_h2 = np.arange(8, 25)
    # neuron_lengths_h3 = np.arange(0, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    # accuracy_scores = np.zeros(len(neuron_lengths_h1)*len(neuron_lengths_h2)*len(neuron_lengths_h3))

    train_single_NN = True

    if train_single_NN == True:
        NN = NeuralNet(X_train, y_train, [32], ['relu'], 'softmax')
        NN.train(iters, gamma, lmbd=lmbd)

        if n_categories == 1:
            y_pred = NN.predict_single_output_neuron(X_test)
        else:
            y_pred = NN.predict(X_test)

        equal = (y_pred == y_test[:,0]).astype(int)
        accuracy = np.mean(equal)
        # exit()
        print ("gamma =", gamma)
        print ("lmbd =", lmbd)
        print ("accuracy =", accuracy)
        print ("--------------\n")

        # ConfMatrix(y_test, y_pred)
        # show_misclassified(X_test, y_test, y_pred)

    # exit()
    config = [32,32]
    NN_grid = NeuralNet(X_train, y_train, config, ['tanh', 'relu'], 'softmax')
    NN_grid.grid_search(X_test, y_test, params, gammas, config)

    # Iterate over multiple hidden layer configurations
    # for i in range(1, 6):
    #     for j in range(0, i+1):
    #         for k in range(0, 1):
    #             config = [i, j, k]
    #             NN_grid.grid_search(X_test, y_test, params, gammas, config)

    best_accuracy, best_config,best_lmbd, best_gamma = NN_grid.return_params()

    print ("\n--- Grid search done ---")
    print ('Best accuracy:', best_accuracy)
    print ("with configuration", best_config, "lmbd =", best_lmbd, "gamma =", best_gamma)


    # scikit-learn NN

    # scikit_NN = MLPClassifier(solver='lbfgs', alpha=0, learning_rate='constant', learning_rate_init=gamma, activation='logistic', hidden_layer_sizes=int(m), random_state=1,max_iter=iters)
    #
    # scikit_NN.fit(X, y[:,0])
    # y_pred = scikit_NN.predict(X)
    # ConfMatrix(y[:,0], y_pred)

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


"""NN"""
# digits dataset
# good configurations:
# [13], lmbd = 0.0001, gamma = 1e-3         accuracy = 0.9777777
# [32,16], lmbd = 0, gamma = 1e-3           accuracy = 0.9777777
# [38], lmbd = 0.001, gamma = 1e-3          accuracy = 0.9805555
# [20, 12], lmbd = 0.001, gamma = 0.001     accuracy = 0.9833333



#
