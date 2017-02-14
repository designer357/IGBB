import os
import numpy as np
import random as rd
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
def cross_tab(data,cross_folder,tab_cv):
    posi_data=data[data[:,-1]!=1.0]
    nega_data=data[data[:,-1]==1.0]
    #print("Positive is "+str(len(posi_data)))
    #print("Negative is "+str(len(nega_data)))

    print("IR is :"+str(float(len(nega_data))/len(posi_data)))
    print("The anomalies is "+str(len(posi_data)))
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
        p_test = posi_data[p_index_test]
        n_test = nega_data[n_index_test]
        #print(nega_data.shape)
        N = len(n_test)
        n_test2 = n_test[N - len(p_test) - 1:N - 1, :]

        test_data = np.append(p_test, n_test, axis=0)

        #Random undersampling without replacement
        train_data = np.concatenate((nega_data[np.random.choice(n_index_train, len(p_train),replace=False)], p_train))

        return train_data[:,:-1],train_data[:,-1],test_data[:,:-1],test_data[:,-1]
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