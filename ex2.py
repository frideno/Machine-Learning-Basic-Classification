# Omri Fridental 323869545
# Yuval Ezra 323830448

import numpy as np
import sys
import train_algorithms


np.seterr(all='ignore')

genderToFloatDict = {b'M': 1, b'F': 2, b'I': 3}

number_of_features = 8
number_of_classes = 3

"""
    zscore normalization - gets set and zscore info (list of (mean, std) per column) and normailize it by that.
"""
def normalize(set, zscore_info):

    m = len(set)
    if m != 0:
        for i, (mean_i, std_i) in zip(range(list(set.shape)[1]), zscore_info):
            if std_i != 0:
                set[:, i] = (set[:, i] - mean_i) / std_i
            else:
                set[:, i] = mean_i / m


"""
    extracting training test from - x values file_name_x, y values file_name_y, and returing it.
    returning also zscore_info (lists of means, lists of stds).
"""
def extractTrain(file_name_x, file_name_y):

    if file_name_x == None or file_name_y == None:
        return None


    # read x set to a matrix  - convert gender string to float.
    xset =  np.loadtxt(file_name_x, delimiter=',',
                       converters={0: lambda gender: genderToFloatDict[gender]})
    # normalize values by zscore.
    column_means_stds = list(zip(xset.mean(axis=0), xset.std(axis=0)))
    normalize(xset, column_means_stds)

    # read y values to a matrix.
    yset = np.loadtxt(file_name_y, delimiter=',', dtype='i4')

    # if the user wanted to extract x, y together:
    set = []
    for x, y in zip(xset, yset):
        set.append((x, y))

    return set, column_means_stds


"""
    extracting test set from file_name_x and normalizing it by zsocre_info.
"""
def extractTest(file_name_x, zscore_info):


    # read x set from x file. normalize values by given means and stds.
    xset = np.loadtxt(file_name_x, delimiter=',',
                           converters={0: lambda gender: genderToFloatDict[gender]})
    normalize(xset, zscore_info)
    return xset


"""   
    predict - given x,w , return y_hat - prediction. by sending back argmax[y](w_y * x)
"""

def predict(x, theta) -> int:

    # theta = wx + b, return the index of which class that maximize wx + b.
    (w, b) = theta
    return np.argmax(np.dot(w, x) + b)



def main():

    # takes file names from argv.
    if len(sys.argv) < 4:
        print('error in args')
        exit(1)

    training_x, training_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]

    # extract training set from files.
    training_set, training_means_stds = extractTrain(training_x, training_y)


    # define algorithms and weights.
    algorithms = []
    algorithms.append(train_algorithms.perceptron(eta=0.0125, T=20))
    algorithms.append(train_algorithms.svm(eta=0.0008, T=20, Lambda=2.0))
    algorithms.append(train_algorithms.pa(eta=0.0001, T=15))
    weights = {}

    # train each algorithm while choosing a hyper parameters
    for algorithm in algorithms:
        weights[str(algorithm)] = algorithm.train(training_set)

    # test:
    test_set = extractTest(file_name_x=test_x, zscore_info=training_means_stds)

    # for each algorithm, create predictions of test set:
    predictions = []
    for algorithmName, theta in weights.items():
        predictions.append([predict(x, theta) for x in test_set])

    # print result of test:
    for predcs in zip(*predictions):
        line = []
        for algorithmName, predictionValue in zip(weights.keys(), predcs):
            line.append(algorithmName + ': ' + str(predictionValue))
        print(', '.join(line))



if __name__== '__main__':
    main()


