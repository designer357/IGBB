import os
import numpy as np
import random as rd
from sklearn.datasets import make_classification

from unbalanced_dataset.unbalanced_dataset import UnbalancedDataset
from unbalanced_dataset.over_sampling import OverSampler
from unbalanced_dataset.over_sampling import SMOTE

from unbalanced_dataset.under_sampling import UnderSampler
from unbalanced_dataset.under_sampling import TomekLinks
from unbalanced_dataset.under_sampling import ClusterCentroids
from unbalanced_dataset.under_sampling import NearMiss
from unbalanced_dataset.under_sampling import CondensedNearestNeighbour
from unbalanced_dataset.under_sampling import OneSidedSelection
from unbalanced_dataset.under_sampling import NeighbourhoodCleaningRule

from unbalanced_dataset.ensemble_sampling import EasyEnsemble
from unbalanced_dataset.ensemble_sampling import BalanceCascade

from unbalanced_dataset.pipeline import SMOTEENN
from unbalanced_dataset.pipeline import SMOTETomek
def loadData(input_data_path,filename):
    global positive_sign,negative_sign
    with open(os.path.join(input_data_path,filename)) as fin:
        data=[]
        for each in fin:
            if '@' in each:
                continue
            val=each.split(",")
            if len(val)>0 or val[-1].strip()=="negative" or val[-1].strip()=="positive":
                if val[-1].strip() == '1.0' or val[-1].strip() == '1':
                    pass
                else:
                    val[-1] = '-1.0'
                try:
                    val=list(map(lambda a:float(a),val))
                except:
                    val=list(map(lambda a:str(a),val))

                val[-1]=int(val[-1])
                data.append(val)
        return np.array(data)
def cross_tab2(input_data_path,filename_training,filename_testing):
    data_testing = loadData(input_data_path,filename_testing)
    data = loadData(input_data_path,filename_training)
    posi_data=data[data[:,-1]!=1.0]
    nega_data=data[data[:,-1]==1.0]
    p_index_train=[i for i in range(len(posi_data))]
    n_index_train=[i for i in range(len(nega_data))]
    p_train = posi_data[p_index_train]
    n_train = nega_data[np.random.choice(n_index_train, len(p_train),replace=False)]
    data_training = np.concatenate((n_train, p_train))
    return data_training[:, :-1], data_training[:, -1], data_testing[:, :-1], data_testing[:, -1]


def cross_tab(data,cross_folder,tab_cv, flag):
    posi_data=data[data[:,-1]!=1.0]
    nega_data=data[data[:,-1]==1.0]
    #print("Positive is "+str(len(posi_data)))
    #print("Negative is "+str(len(nega_data)))

    #print("IR is :"+str(float(len(nega_data))/len(posi_data)))
    #print("The anomalies is "+str(len(posi_data)))
    p_indexes=[i for i in range(len(posi_data))]
    n_indexes=[i for i in range(len(nega_data))]
    for tab_cross in range(cross_folder):
        if not tab_cross == tab_cv:continue
        p_index_train = []
        p_index_test = []
        n_index_train = []
        n_index_test = []

        for tab_positive in p_indexes:
            if int((cross_folder - tab_cross - 1) * len(posi_data) / cross_folder) < tab_positive < int(
                                    (cross_folder - tab_cross) * len(posi_data) / cross_folder):
                p_index_test.append(tab_positive)
            else:
                p_index_train.append(tab_positive)
        for tab_negative in n_indexes:
            if int((cross_folder - tab_cross - 1) * len(nega_data) / cross_folder) < tab_negative < int(
                                    (cross_folder - tab_cross) * len(nega_data) / cross_folder):
                n_index_test.append(tab_negative)
            else:
                n_index_train.append(tab_negative)


        p_train = posi_data[p_index_train]
        n_train = nega_data[n_index_train]

        p_test = posi_data[p_index_test]
        n_test = nega_data[n_index_test]
        #print(nega_data.shape)
        N = len(n_test)
        n_test2 = n_test[N - len(p_test) - 1:N - 1, :]

        test_data = np.append(p_test, n_test, axis=0)
        testX = test_data[:,:-1]
        testY = test_data[:,-1]

        train_data = np.concatenate((n_train, p_train))

        #Random undersampling without replacement
        if flag == 'RUS':
            train_data = np.concatenate((nega_data[np.random.choice(n_index_train, len(p_train),replace=False)], p_train))
            trainX = train_data[:,:-1]
            trainY = train_data[:,-1]
        elif flag == "SMOTE":
            sm_ = SMOTE(kind='regular', verbose=True)
            trainX, trainY = sm_.fit_transform(train_data[:,:-1], train_data[:,-1])
        elif flag == "ROS":
            os_ = OverSampler(verbose=True)
            trainX, trainY = os_.fit_transform(train_data[:,:-1], train_data[:,-1])
        elif flag == "NCL":
            ncr_ = NeighbourhoodCleaningRule(verbose=True)
            trainX, trainY = ncr_.fit_transform(train_data[:,:-1], train_data[:,-1])
        elif flag == "USCC":
            cc_ = ClusterCentroids(verbose=True)
            trainX, trainY = cc_.fit_transform(train_data[:,:-1], train_data[:,-1])
        return trainX,trainY,testX,testY
"""
input_data_path = os.path.join(os.getcwd(),"BGPData")
filename = "HB_Nimda.txt"
data = loadData(input_data_path,filename)
print(data.shape)
trainX,trainY,testX,testY = cross_tab(data,2,1)
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)
"""