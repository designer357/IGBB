import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
#from sklearn.datasets import make_hastie_10_2
import os
import loaddata
import visualize
def entropyVector(x):
    '''
    Computes the entropy of a vector of discrete values
    '''
    vals = np.bincount(x)
    # excluding the empty bins at the end and start of x
    vals = vals[1:-2]
    den = np.sum(vals)
    probs = vals / np.float(den)
    entro = -np.sum([i * np.log2(i) for i in probs if i != 0]) / np.float(np.log2(np.size(vals, 0)))
    #     plt.plot(x)
    #     plt.show()
    return entro


def mutualInformation(jointProb):
    '''
    Calculates the mutual information from the output of
    the jointProbs function
    '''
def jointProbs(x, y):
    '''
    Calculates the probabilities of joint probabilities for the different values of x and y
    The values most be discrete,positive, starting from 0 or 1
    '''
    probs = {}

    valsX, temp = np.unique(x, return_inverse=True)
    valsX = [int(i) for i in valsX]
    temp = np.bincount(temp)
    pX = temp / float(sum(temp))

    valsY, temp = np.unique(y, return_inverse=True)
    valsY = [int(i) for i in valsY]
    temp = np.bincount(temp)
    pY = temp / float(sum(temp))

    maxX = max(valsX)
    maxY = max(valsY)

    minX = min(valsX)
    minY = min(valsY)

    C = {}

    # Another option would be to use sparse matrices however that will
    # only work for 2 dimensional matrices

    # This is a very efficient version of the algorithm
    for i in range(len(x)):
        key = str(x[i]) + ',' + str(int(y[i]))
        if key in C.keys():
            C[key] += 1
        else:
            C[key] = 1

    den = 0
    for xi in valsX:
        for yi in valsY:
            key = str(xi) + ',' + str(yi)
            if key in C.keys():
                probs[key] = C[key]
                den += C[key]
            else:
                probs[key] = 0

    for (key, val) in probs.iteritems():
        probs[key] = probs[key] / float(den)

    totalSum = 0
    for key, val in probs.iteritems():
        xVal, yVal = [int(i) for i in key.split(',')]

        indX = valsX.index(xVal)
        indY = valsY.index(yVal)
        if probs[key] == 0:
            totalSum += 0
        else:
            totalSum += probs[key] * np.log(probs[key] / (pX[indX] * pY[indY]))

            #     return {'counts':C,'probabilities':probs}
    return totalSum


def informationGain(x, y):
    '''
    This implementation of information gain
    simply binerizes the data and then
    calculates information gain following the
    elements of information theory book equation in page
    x : features to be analyzed, where rows are data points and columns correspond to features
    y : class labels
    here Information gain is:
    I(y;x)=H(y)-H(y|x)
    I(y;x)=\sum_{x,y} p(x,y)\frac {log(p(x,y))}{(p(x).p(y))}
    '''

    x = np.array(x)
    y = np.array(y)

    xMax = np.max(x, 0)
    xMin = np.min(x, 0)
    bins = []

    # Discretized data
    xD = []
    try:
        for i in range(len(xMin)):
            # -1e-5 and+1e-5 added so that the extremes are included in the
            # bins, this however generates two empty bins that's why the last
            # [1:-2]
            tempXD, tempBins = np.histogram(x[:, i], 10)
            xD.append(np.digitize(x, tempBins))
            bins.append(tempBins)
            C = jointProbs(xD[-1], y)
            print(C)
    except:
        #         bins.append(np.linspace(xMin-1e-5,xMax+1e-5, num=10))
        binsNum = len(np.unique(x))
        if binsNum < 10:
            temp, tempBins = np.histogram(x, binsNum)
            xD.append(np.digitize(x, tempBins))
        else:
            temp, tempBins = np.histogram(x, 10)
            xD.append(np.digitize(x, tempBins))
        C = jointProbs(xD[-1], y)
    return C
def top_feat(data,label,w,k):
    new_data = np.multiply(data.T,w).T
    top_list=[]
    for tab in range(len(data[0])):
        top_list.append(informationGain(new_data[:, tab], label))
    result=(sorted(enumerate(top_list),key=lambda a:a[1],reverse=True))
    label_=[e[0] for e in result]
    return label_[:k]
def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))

def print_error_rate(err):
    print 'Error rate: Training: %.4f - Test: %.4f' % err

def generic_clf(clf, trainX,trainY, testX, testY):
    clf.fit(trainX, trainY)
    pred_train = clf.predict(trainX)
    pred_test = clf.predict(testX)
    return get_error_rate(pred_train, trainY), \
           get_error_rate(pred_test, testY)


"""==================== Ada.Boost Implementation ======================"""


def igboost_clf(clf, M, trainX, trainY, testX, testY, using_weights=True):
    global top_k
    n_train, n_test = len(trainX), len(testX)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for i in range(M):
        # Fit a classifier with the specific weights
        if i > 0 and using_weights == True:
            top_features = top_feat(trainX,trainY,w,top_k)
            trainX = trainX[:,top_features]
            testX = testX[:,top_features]
        else:
            pass
        clf.fit(trainX, trainY, sample_weight=w)
        pred_train_i = clf.predict(trainX)
        pred_test_i = clf.predict(testX)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != trainY)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x == 1 else -1 for x in miss]
        # Error
        err_m = np.dot(w, miss) / sum(w)
        # Alpha
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train,
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test,
                                         [x * alpha_m for x in pred_test_i])]

    return np.sign(pred_test)
    #pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # Return error rate in train and test set
    #return get_error_rate(pred_train, trainY), \
           #get_error_rate(pred_test, testY)


"""==================== MainFunc ======================"""
if __name__ == '__main__':

    # Read data
    #x, y = make_gaussian_quantiles()
    #df = pd.DataFrame(x)
    #df['Y'] = y

    global top_k,max_n,interval_n
    top_k = 15
    max_n = 400
    interval_n = 10
    input_data_path = os.path.join(os.getcwd(), "BGPData")
    filename = "HB_Nimda.txt"
    data = loaddata.loadData(input_data_path, filename)
    print(data.shape)
    X_train, Y_train2, X_test, Y_test = loaddata.cross_tab(data, 2, 1)
    # Split into training and test set
    #train, test = train_test_split(df, test_size=0.2)
    #X_train, Y_train = train.ix[:, :-1], train.ix[:, -1]
    #X_test, Y_test = test.ix[:, :-1], test.ix[:, -1]
    print(X_train.shape)
    #X_train = X_train.values
    #X_test = X_test.values
    # Fit a simple decision tree first
    clf_tree = DecisionTreeClassifier(max_depth=2, random_state=1)
    er_tree = generic_clf(clf_tree,X_train,Y_train2, X_test, Y_test)

    # Fit Adaboost classifier using a decision tree as base estimator
    # Test with different number of iterations
    #er_train, er_test1 = [er_tree[0]], [er_tree[1]]
    #er_train, er_test2 = [er_tree[0]], [er_tree[1]]
    er_train = []
    er_test1 = []
    er_test2 = []
    x_range = range(10, max_n, interval_n)
    for i in x_range:
        print(str(i)+"_th of not using weights is running...")
        er_i = igboost_clf(clf_tree, i, X_train, Y_train2, X_test, Y_test,False)
        er_train.append(er_i[0])
        er_test1.append(er_i[1])
    for i in x_range:
        print(str(i)+"_th of using weights and IG is running...")
        er_i = igboost_clf(clf_tree, i, X_train, Y_train2, X_test, Y_test)
        er_train.append(er_i[0])
        er_test2.append(er_i[1])
    # Compare error rate vs number of iterations
    visualize.plot_error_rate(er_test1, er_test2)