# Omri Fridental 323869545
# Yuval Ezra 323830448
import numpy as np

number_of_features = 8
number_of_classes = 3


"""
    perceptron algorithm class.
    paramerts: eta, number_of_epochs

"""

class perceptron:
    def __init__(self, eta, T):
        self.eta = eta
        self.T = T

    def train(self, training_set):

        # initialize w,b, eta:
        w = np.zeros((number_of_classes, number_of_features))
        b = np.full(number_of_classes, 1.0)
        eta = self.eta

        training_set_copy = training_set[:]
        # perceptron iteration:
        for epoch in range(self.T):

            # suffle the training_set_copy, and iterate its members:
            np.random.shuffle(training_set_copy)
            for x, y in training_set_copy:

                # try to classify x, and if fails, alter w by it:
                y_hat = np.argmax(np.dot(w, x) + b)
                if y_hat != y:
                    w[y] = w[y] + eta * x
                    b[y] = b[y] + eta

                    w[y_hat] = w[y_hat] - eta * x
                    b[y_hat] = b[y_hat] - eta

            # updating learning rate to make convergence faster.
            eta = eta / (epoch + 1)

        return (w, b)

    def __str__(self):
        return 'perceptron'


"""
    perceptron algorithm class.
    paramerts: eta, number_of_epochs

"""

class svm:
    def __init__(self, eta, T, Lambda):
        self.eta = eta
        self.T = T
        self.Lambda = Lambda

    def train(self, training_set):

        # initialize w,b , eta:
        w = np.zeros((number_of_classes, number_of_features))
        b = np.full((number_of_classes), 1.0)
        eta = self.eta
        Lambda = self.Lambda

        training_set_copy = training_set[:]

        # perceptron iteration:
        for epoch in range(self.T):

            # suffle the training_set_copy, and iterate its members:
            np.random.shuffle(training_set_copy)
            bad_y_hat = 0

            for x, y in training_set_copy:

                # try to predict, and if wrong alter w.
                y_hat = np.argmax(np.dot(w, x) + b)
                if y_hat != y:

                    # update rule of svm.
                    for i in range(0, number_of_classes):
                        w[i] *= (1 - eta * Lambda)
                        b[i] *= (1 - eta * Lambda)
                    w[y] += eta * x
                    w[y_hat] -= eta * x
                    b[y] += eta
                    b[y_hat] -= eta

                    bad_y_hat += 1

            # updating eta to faster the convergence.
            eta = eta / (epoch + 1)
            Lambda += 1

        return (w, b)

    def __str__(self):
        return 'svm'



class pa:
    def __init__(self, eta, T):
        self.eta = eta
        self.T = T

    def train(self, training_set):

        # initialize w, b:
        w = np.zeros((number_of_classes, number_of_features))
        b = np.full(number_of_classes, 1)
        eta = self.eta

        training_set_copy = training_set[:]

        # perceptron iteration - with T epochs
        for epoch in range(self.T):

            # suffle the training_set_copy, and iterate its members:
            np.random.shuffle(training_set_copy)

            for x, y in training_set_copy:

                # try to predict and if fails alter w.
                y_hat = np.argmax(np.dot(w, x) + b)

                if y_hat != y:
                    Tau = max(0, 1 - np.dot(w[y], x) + np.dot(w[y_hat], x)) / (np.linalg.norm(x) ** 2)
                    w[y] += eta * Tau * x
                    w[y_hat] -= eta * Tau * x
                    b[y] += eta * Tau
                    b[y_hat] -= eta * Tau

            eta /= (epoch + 1)

        return (w, b)

    def __str__(self):
        return 'pa'
