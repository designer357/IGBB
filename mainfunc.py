#_author_by_MC@20160424
import os
import time
import math
import evaluation
start = time.time()
import numpy as np
import loaddata
import random as RANDOM
#from svmutil import *
#import seaborn as sns
#import matplotlib.pyplot as plt
from numpy import *
from sklearn import tree
from InformationGain import *
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA,KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm,datasets,preprocessing,linear_model
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier
def entropyVector(x):
    #Computes the entropy of a vector of discrete values
    vals = np.bincount(x)
    vals = vals[1:-2]
    den = np.sum(vals)
    probs = vals / np.float(den)
    entro = -np.sum([i * np.log2(i) for i in probs if i != 0]) / np.float(np.log2(np.size(vals, 0)))
    return entro


def mutualInformation(jointProb):
    '''
    Calculates the mutual information from the output of
    the jointProbs function
    '''
def jointProbs(x, y):

    #Calculates the probabilities of joint probabilities for the different values of x and y
    #The values most be discrete,positive, starting from 0 or 1

    probs = {}
    valsX, temp = np.unique(x, return_inverse=True)
    valsX = [int(i) for i in valsX]
    temp = np.bincount(temp)
    pX = temp / float(sum(temp))

    valsY, temp = np.unique(y, return_inverse=True)
    valsY = [int(i) for i in valsY]
    temp = np.bincount(temp)
    pY = temp / float(sum(temp))

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

    return totalSum


def informationGain(x, y):
    '''
    This implementation of information gain
    x : features to be analyzed, where rows are data points and columns correspond to features
    y : class labels
    Information gain is:I(y;x)=H(y)-H(y|x);I(y;x)=\sum_{x,y} p(x,y)\frac {log(p(x,y))}{(p(x).p(y))}
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
            tempXD, tempBins = np.histogram(x[:, i], 10)
            xD.append(np.digitize(x, tempBins))
            bins.append(tempBins)
            C = jointProbs(xD[-1], y)
            print(C)
    except:
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
    return get_error_rate(pred_train, trainY), get_error_rate(pred_test, testY)


def igboost_clf(clf, M, top_k, trainX, trainY, testX, testY, using_weights=True):
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

    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # Return error rate in train and test set
    return pred_test
    #return get_error_rate(pred_train, trainY), get_error_rate(pred_test, testY)

def MainFunc(method_label,bagging_size,input_data_path,filename):
    global positive_sign,negative_sign,boosting_i, top_k
    #print(filename+" is processing......")
    data = loaddata.loadData(input_data_path, filename)
    voting_list = [[] for i in range(bagging_size)]
    output=[]
    for bagging_number in range(bagging_size):
        print("The Bagging Number is " + str(bagging_number+1) + "...")
        X_train, Y_train, X_test, Y_test = loaddata.cross_tab(data, 2, 1)
        if method_label == 0:
            result = igboost_clf(DecisionTreeClassifier(max_depth=2, random_state=1), boosting_i,top_k, X_train, Y_train, X_test, Y_test)
        elif method_label==1:
            #clf = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.1,max_depth=2)
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, random_state=1))
            clf.fit(X_train, Y_train)
            result = clf.predict(X_test)
            #classifier = train(X_Training,Y_Training,Top_K)
        elif method_label==2:
            clf=tree.DecisionTreeClassifier(max_depth=2, random_state=1)
            clf.fit(X_train, Y_train)
            result = clf.predict(X_test)
        elif method_label==3:
            scaler = preprocessing.StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            clf = svm.SVC(kernel="rbf", gamma=0.001)
            clf.fit(X_train, Y_train)
            result = clf.predict(X_test)
        elif method_label==4:
            clf = linear_model.LogisticRegression()
            clf.fit(X_train, Y_train)
            result = clf.predict(X_test)
        elif method_label==5:
            clf = KNeighborsClassifier(5)
            clf.fit(X_train, Y_train)
            result = clf.predict(X_test)

        voting_list[bagging_number].extend(result)

    temp = np.array(voting_list).T
    ac_positive = 0
    ac_negative = 0
    for tab_i in range(len(temp)):
        if list(temp[tab_i]).count(positive_sign)>list(temp[tab_i]).count(negative_sign):
            output.append(positive_sign)
        else:
            output.append(negative_sign)

    for tab in range(len(output)):
        if output[tab]==positive_sign and output[tab]==int(Y_test[tab]):
            ac_positive += 1
        if output[tab]==negative_sign and output[tab]==int(Y_test[tab]):
            ac_negative += 1
    g_mean=np.sqrt(float(ac_positive*ac_negative)/(list(Y_test).count(positive_sign)*list(Y_test).count(negative_sign)))
    #auc=float(get_auc(np.array(Output),Y_Testing,positive_sign))
    print(Y_test)
    print(np.array(output))

    auc=float(roc_auc_score(Y_test, np.array(output)))
    #print("TP is..."+str(float(ac_positive)/Output.count(positive_sign)))
    #print("TN is..."+str(float(ac_negative)/Output.count(negative_sign)))
    accuracy = float(ac_positive+ac_negative)/len(output)

    return g_mean,auc,accuracy



if __name__=='__main__':
    global positive_sign,negative_sign,boosting_i, top_k
    #os.chdir("/home/grads/mcheng223/IGBB")
    positive_sign = -1
    negative_sign = 1
    count_positive= 0
    count_negative= 0
    boosting_i = 5
    top_k = 15
    bagging_size = 3
    input_data_path = os.path.join(os.getcwd(),"BGPData")

    out_put_path = os.path.join(os.getcwd(),"Output_BGPData")
    if not os.path.isdir(out_put_path):
        os.makedirs(out_put_path)
    filenamelist=os.listdir(input_data_path)

    #Method_Dict={"DT":1,"LR":4}
    #method_dict={"IGBB":0,"AdaBoost":1,"DT":2,"SVM":3,"LR":4,"KNN":5}
    method_dict={"AdaBoost":1}

    for eachfile in filenamelist:
        if 'HB_Nimda' in eachfile and '.txt' in eachfile:
            if 'Multi' in eachfile:continue
            else:
                pass
        else:
            continue
        for eachMethod,eachMethodLabel in method_dict.items():
            print(eachMethod + " is running...")
            g_mean,auc,accuracy = MainFunc(eachMethodLabel,bagging_size, input_data_path,eachfile)
            print("The G_mean of "+eachMethod+ " is "+str(g_mean))
            print("The AUC of "+eachMethod+ " is "+str(auc))
            print("The Accuracy of "+eachMethod+ " is "+str(accuracy))


    print(time.time()-start)