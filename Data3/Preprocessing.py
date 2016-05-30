import os
import time
import math
start = time.time()
import numpy as np
import random as RANDOM
from svmutil import *
#import seaborn as sns
#import matplotlib.pyplot as plt
from numpy import *
from sklearn import tree
from InformationGain import *
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA,KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm,datasets,preprocessing,linear_model
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier
positive_sign=-1
negative_sign=1
count_positive=0
count_negative=0
def LoadData(filename):
    global input_data_path,out_put_path

    #y_svmformat, x_svm_format = svm_read_problem(os.path.join(os.getcwd(),"AS_Path_Error_half_minutes.txt"))
    #y_svmformat, x_svm_format = svm_read_problem(os.path.join(os.getcwd(),"AS_Leak_half_minutes.txt"))
    #y_svmformat, x_svm_format = svm_read_problem(os.path.join(os.getcwd(),"AS_Filtering_Error_half_minutes.txt"))
    y_svmformat, x_svm_format = svm_read_problem(os.path.join(input_data_path,filename))
    #y_svmformat, x_svm_format = svm_read_problem(os.path.join(os.getcwd(),"Nimda_AS_513_half_minutes.txt"))
    #y_svmformat, x_svm_format = svm_read_problem(os.path.join(os.getcwd(),"Code_Red_I_AS_513_half_minutes.txt"))
    #y_svmformat, x_svm_format = svm_read_problem(os.path.join(os.getcwd(),"AS_Filtering_Error_AS_286_half_minutes.txt"))

    y_svmformat=np.array(y_svmformat)
    y_svmformat[y_svmformat==-1]=positive_sign#Positive is -1

    Data=[]
    for tab in range(len(x_svm_format)):
        Data.append([])
        temp=[]
        for k,v in x_svm_format[tab].items():
            temp.append(float(v))
        Data[tab].extend(temp)
        Data[tab].append(int(y_svmformat[tab]))
    Data=np.array(Data)
    print(len(y_svmformat[y_svmformat==-1]))
    #np.random.shuffle(Data)
    #np.random.shuffle(Data)
    print(len(y_svmformat[y_svmformat==1]))
    Positive_Data_Samples=Data[Data[:,-1]==positive_sign]
    Negative_Data_Samples=Data[Data[:,-1]==negative_sign][:len(Positive_Data_Samples)]
    TrainingSamples=np.concatenate((Negative_Data_Samples,Positive_Data_Samples))
    Positive_Data = TrainingSamples[TrainingSamples[:,-1]==positive_sign]
    Negative_Data = TrainingSamples[TrainingSamples[:,-1]==negative_sign]
    #print(len(Positive_Data_Samples))
    print("IR is :"+str(float(len(Negative_Data))/len(Positive_Data)))
    print(TrainingSamples[0])
    print(TrainingSamples[1])
    print(TrainingSamples[2])

    with open(filename+".txt","w")as fout:
        for tab1 in range(len(TrainingSamples)):
            for tab2 in range(len(TrainingSamples[0])-1):
                fout.write(str(TrainingSamples[tab1][tab2]))
                fout.write(',')
            fout.write(str(TrainingSamples[tab1][tab2+1]))
            fout.write('\n')

    return TrainingSamples
input_data_path = os.getcwd()
LoadData("BGP_DATA")
#LoadData("Code_Red_I")
#LoadData("Nimda")
#LoadData("Slammer")