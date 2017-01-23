#_author_by_MC@20160424
import os
import time
import math
start = time.time()
import numpy as np
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA,KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm,datasets,preprocessing,linear_model
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier,BaggingClassifier
global Classifier_Label
Classifier_Label=None
def LoadData(input_data_path,filename):
    global count_positive,count_negative,Classifier_Label
    with open(os.path.join(input_data_path,filename)) as fin:
        if filename == 'sonar.dat':
            negative_flag = 'M'
        elif filename == 'bands.dat':
            negative_flag = 'noband'
        elif filename =='Ionosphere.dat':
            negative_flag = 'g'
        elif filename =='spectfheart.dat':
            negative_flag = 'g'
        elif filename =='spambase.dat':
            negative_flag = '0'
        elif filename =='page-blocks0.dat':
            negative_flag = 'negative'
        elif filename =='blocks0.dat':
            negative_flag = 'g'
        elif filename =='heart.dat':
            negative_flag = '2'
        elif filename =='segment0.dat':
            negative_flag = 'g'
        else:
            negative_flag = '1.0'
        Data=[]
        for each in fin:
            if '@' in each:
                continue
            val=each.split(",")
            if len(val)>0 or val[-1].strip()=="negative" or val[-1].strip()=="positive":
                if val[-1].strip()== negative_flag:
                    val[-1]= 1.0
                    count_negative += 1
                else:
                    if Classifier_Label == "Multi":
                        pass
                    else:
                        val[-1]=positive_sign
                    count_positive += 1
                try:
                    val=map(lambda a:float(a),val)
                except:
                    val=map(lambda a:str(a),val)

                val[-1]=int(val[-1])
                Data.append(val)
        Data=np.array(Data)
        return Data


# Building weak stump function
def buildWeakStump(d,l,D,Sub_Features):
    d2=d[:,Sub_Features]
    dataMatrix = mat(d2)
    labelmatrix = mat(l).T
    m,n = shape(dataMatrix)
    numstep = 10.0
    bestStump = {}
    bestClass = mat(zeros((5,1)))
    minErr = inf
    for i in range(n):
        datamin = dataMatrix[:,i].min()
        datamax = dataMatrix[:,i].max()
        stepSize = (datamax - datamin) / numstep
        for j in range(-1,int(numstep)+1):
            for inequal in ['lt','gt']:
                threshold = datamin + float(j) * stepSize
                predict = stumpClassify(dataMatrix,i,threshold,inequal)
                err = mat(ones((m,1)))
                err[predict == labelmatrix] = 0.0
                weighted_err = D.T * err
                if weighted_err < minErr:
                    minErr = weighted_err
                    bestClass = predict.copy()
                    bestStump['dim'] = i
                    bestStump['threshold'] = threshold
                    bestStump['ineq'] = inequal
    return bestStump, minErr, bestClass

# Use the weak stump to classify training data
def stumpClassify(datamat,dim,threshold,inequal):
    res = ones((shape(datamat)[0],1))
    if inequal == 'lt':
        res[datamat[:,dim] <= threshold] = positive_sign
    else:
        res[datamat[:,dim] > threshold] = negative_sign

    return res


def Return_Top_K_Features(data,label,W,K):
    Features=[i for i in range(len(data[0]))]
    data_copy=data.copy()
    y_=label
    Top_List=[]

    for tab in range(len(Features)):

        if len(data_copy[:,tab])==len(W):
            for i in range(len(data_copy[:,tab])):
                data_copy[:,tab][i]=W[i]*data_copy[:,tab][i]
        else:
            print("Error! Data_[Column] Not Equal to Weight")


        Top_List.append(informationGain(data_copy[:,tab],y_))

    result=(sorted(enumerate(Top_List),key=lambda a:a[1],reverse=True))
    Label=[e[0] for e in result]
    return Label[:K]


# Training
def train(data,label,Top_K,numIt = 1000,flag = 0):
    SubSpace_WeakClassifiers={"weakClassifiers":[],"subSpace":[]}
    #weakClassifiers = []
    m = shape(data)[0]
    D = mat(ones((m,1))/m)

    Sub_Features=sorted(Return_Top_K_Features(data,label,D,Top_K))

    EnsembleClassEstimate = mat(zeros((m,1)))
    Sub_Features_List=[]
    for i in range(numIt):
        #print("The "+str(i)+" th iterations...")

        bestStump, error, classEstimate = buildWeakStump(data,label,D,Sub_Features)
        #print("Error is -------------------"+str(error))
        alpha = float(0.5*log((1.0-error) / (error+1e-15)))
        bestStump['alpha'] = alpha
        #weakClassifiers.append(bestStump)
        SubSpace_WeakClassifiers["weakClassifiers"].append(bestStump)
        weightD = multiply((-1*alpha*mat(label)).T,classEstimate)
        D = multiply(D,exp(weightD))
        D = D/D.sum()
        EnsembleClassEstimate += classEstimate*alpha
        EnsembleErrors = multiply(sign(EnsembleClassEstimate)!=mat(label).T,\
                                  ones((m,1)))  #Converte to float
        errorRate = EnsembleErrors.sum()/m
        #print "total error:  ",errorRate
        if errorRate == 0.0:
            break
        Sub_Features=sorted(Return_Top_K_Features(data,label,D,Top_K))
        SubSpace_WeakClassifiers["subSpace"].append([Sub_Features])

        #if not flag==0:
            #for each_feature in Sub_Features:
                #Sub_Features_List.append(str(each_feature))
            #Sub_Features_List.append('\n')
    #if not flag==0:
        #with open("Sub_Feature_List.txt","w")as fout:
            #for each in Sub_Features_List:
                #fout.write(each)
    print("Complete...")
    #print(SubSpace_WeakClassifiers)
    return SubSpace_WeakClassifiers


# Applying adaboost classifier for a single data sample
def adaboostClassify(dataTest,classifier):
    dataMatrix = mat(dataTest)
    m = shape(dataMatrix)[0]
    EnsembleClassEstimate = mat(zeros((m,1)))
    for i in range(len(classifier["weakClassifiers"])):
        Temp = dataTest[classifier["subSpace"][i]]
        classEstimate = stumpClassify(mat(Temp),classifier["weakClassifiers"][i]['dim'],classifier["weakClassifiers"][i]['threshold'],classifier["weakClassifiers"][i]['ineq'])
        EnsembleClassEstimate += classifier["weakClassifiers"][i]['alpha']*classEstimate
        #print EnsembleClassEstimate
    return sign(EnsembleClassEstimate)

# Testing
def test(dataSet,classifier):
    label = []
    #print '\n\n\nResults: '
    for i in range(shape(dataSet)[0]):
        label.append(adaboostClassify(dataSet[i,:],classifier))
        #print('%s' %(label[0]))
    #print(label)
    return label

def get_auc(arr_score, arr_label, pos_label):
    score_label_list = []
    for index in xrange(len(arr_score)):
        score_label_list.append((float(arr_score[index]), int(arr_label[index])))
    score_label_list_sorted = sorted(score_label_list, key = lambda line:line[0], reverse = True)

    fp, tp = 0, 0
    lastfp, lasttp = 0, 0
    A = 0
    lastscore = None

    for score_label in score_label_list_sorted:
        score, label = score_label[:2]
        if score != lastscore:
            A += trapezoid_area(fp, lastfp, tp, lasttp)
            lastscore = score
            lastfp, lasttp = fp, tp
        if label == pos_label:
            tp += 1
        else:
            fp += 1

    A += trapezoid_area(fp, lastfp, tp, lasttp)
    A /= (fp * tp)
    return A

def trapezoid_area(x1, x2, y1, y2):
    delta = abs(x2 - x1)
    return delta * 0.5 * (y1 + y2)

def Compute_average_list(mylist):
    temp = 0
    for i in range(len(mylist)):
        temp += float(mylist[i])
    return float(temp)/len(mylist)
#def InformationGainBoosting(Iterations):
def Main(Method_Dict,filename):
    #Name_Str_List = ["Code_Red_I_NimdaSlammer.txt","Code_Red_I_SlammerNimda.txt","Nimda_SlammerCode_Red_I.txt"]

    global input_data_path,out_put_path,Classifier_Label
    Classifier_Label = "Multi"
    print(filename+" is processing......")

    Data_=LoadData(input_data_path,filename)
    np.random.shuffle(Data_)
    #Positive_Data=Data_[Data_[:,-1]==positive_sign]
    #Negative_Data=Data_[Data_[:,-1]==negative_sign]
    Positive_Data=Data_[Data_[:,-1]!=1.0]
    Negative_Data=Data_[Data_[:,-1]==1.0]
    print(Data_[:,-1])
    print("IR is :"+str(float(len(Negative_Data))/len(Positive_Data)))
    count_positive = len(Positive_Data)
    count_negative = len(Negative_Data)
    cross_folder=3
    Positive_Data_Index_list=[i for i in range(len(Positive_Data))]
    Negative_Data_Index_list=[i for i in range(len(Negative_Data))]

    Method_List=[k for k,v in Method_Dict.items()]
    Plot_auc_list=[]
    Plot_g_mean_list=[]
    Auc_list = {}
    ACC_list = {}
    G_mean_list = {}
    Temp_Bagging_Auc_list = {}
    Temp_Bagging_ACC_list = {}
    Temp_Bagging_G_mean_list = {}
    Deviation_list={}
    Temp_SubFeature_Auc_list = {}
    Temp_SubFeature_G_mean_list = {}
    Temp_SubFeature_ACC_list = {}

    for eachMethod,methodLabel in Method_Dict.items():
        print(eachMethod+" is running...")
        Auc_list[eachMethod] = []
        ACC_list[eachMethod] = []
        G_mean_list[eachMethod] = []
        Top_K_List = []
        Total_Dimensions = len(Positive_Data[0])

        #for iteration_count in range(10):
        for bagging_number in range(10,100,10):
            print("The Bagging Number is "+str(bagging_number)+"...")
            Temp_Bagging_ACC_list[eachMethod+"_BN_"+str(bagging_number)] = []
            Temp_Bagging_Auc_list[eachMethod+"_BN_"+str(bagging_number)] = []
            Temp_Bagging_G_mean_list[eachMethod+"_BN_"+str(bagging_number)] = []
            Iterations = 1
            for Top_K in range(Total_Dimensions,Total_Dimensions+1,2):
                Temp_SubFeature_ACC_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] = []
                Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] = []
                Temp_SubFeature_G_mean_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] = []
                Deviation_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)] =[]
                print("The Top_K is :"+str(Top_K))
                Top_K_List.append(Top_K)

                for iteration_count in range(Iterations):
                    print(str(iteration_count+1)+"th iterations is running...")
                    cross_folder_auc_list=[]
                    cross_folder_acc_list=[]
                    cross_folder_g_mean_list=[]
                    for tab_cross in range(cross_folder):
                        if not tab_cross == 1: continue
                        Positive_Data_Index_Training=[]
                        Positive_Data_Index_Testing=[]
                        Negative_Data_Index_Training=[]
                        Negative_Data_Index_Testing=[]

                        for tab_positive in Positive_Data_Index_list:
                            if int((cross_folder-tab_cross-1)*len(Positive_Data)/cross_folder)<tab_positive<int((cross_folder-tab_cross)*len(Positive_Data)/cross_folder):
                                Positive_Data_Index_Testing.append(tab_positive)
                            else:
                                Positive_Data_Index_Training.append(tab_positive)
                        for tab_negative in Negative_Data_Index_list:
                            if int((cross_folder-tab_cross-1)*len(Negative_Data)/cross_folder)<tab_negative<int((cross_folder-tab_cross)*len(Negative_Data)/cross_folder):
                                Negative_Data_Index_Testing.append(tab_negative)
                            else:
                                Negative_Data_Index_Training.append(tab_negative)

                        Positive_Training_Data=np.array(Positive_Data)[Positive_Data_Index_Training]
                        Positive_Testing_Data=np.array(Positive_Data)[Positive_Data_Index_Testing]
                        Negative_Training_Data=np.array(Negative_Data)[Negative_Data_Index_Training]
                        Negative_Testing_Data=np.array(Negative_Data)[Negative_Data_Index_Testing]
                        Training_Data = np.concatenate((Negative_Training_Data, Positive_Training_Data))
                        N = len(Negative_Testing_Data)
                        Negative_Testing_Data2 = Negative_Testing_Data[N-len(Positive_Testing_Data)-1:N-1,:]
                        Testing_Data=np.append(Positive_Testing_Data,Negative_Testing_Data2,axis=0)
                        #import matplotlib.pyplot as plt
                        #plt.plot(Negative_Data_Index_Testing,Negative_Testing_Data,'g')
                        #plt.plot(Positive_Data_Index_Testing,Positive_Testing_Data,'r')

                        #plt.show()

                        #Features=[i for i in range(len(Positive_Training_Data[0])-1)]
                        #Sub_Features=Features[:Top_K]
                        N = len(Negative_Testing_Data)
                        Negative_Testing_Data2 = Negative_Testing_Data[N-len(Positive_Testing_Data)-1:N-1,:]
                        Testing_Data=np.append(Positive_Testing_Data,Negative_Testing_Data,axis=0)

                        X_Training=Training_Data[:, :-1]
                        Y_Training=Training_Data[:, -1]

                        X_Testing=Testing_Data[:, :-1]
                        Y_Testing=Testing_Data[:, -1]

                        if methodLabel == 1:
                            # clf = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.1,max_depth=2)
                            baseleaner = AdaBoostClassifier()
                            # clf = train(X_Training,Y_Training,Top_K)
                            # TempList=test(X_test,classifier)
                        elif methodLabel == 2:
                            baseleaner = tree.DecisionTreeClassifier()
                        elif methodLabel == 3:
                            baseleaner = svm.SVC(kernel="rbf", gamma=0.001, C=1000)
                        elif methodLabel == 4:
                            baseleaner = linear_model.LogisticRegression()
                        elif methodLabel == 5:
                            baseleaner = KNeighborsClassifier(3)
                        elif methodLabel==6:
                            baseleaner = AdaBoostClassifier()


                        if methodLabel == 1:
                            D = [1 / float(len(Y_Training)) for i in range(len(Y_Training))]
                            Sub_Features = sorted(Return_Top_K_Features(X_Training, Y_Training, D, 32))
                            X_Training = X_Training[:, Sub_Features]
                            X_Testing = X_Testing[:, Sub_Features]
                        if methodLabel == 6:
                            D = [1 / float(len(Y_Training)) for i in range(len(Y_Training))]
                            Sub_Features = sorted(Return_Top_K_Features(X_Training, Y_Training, D, 33))
                            X_Training = X_Training[:, Sub_Features]
                            X_Testing = X_Testing[:, Sub_Features]

                        if methodLabel == 4:
                            scaler = preprocessing.StandardScaler()
                            X_Training = scaler.fit_transform(X_Training)
                            X_Testing = scaler.fit_transform(X_Testing)

                        clf = BaggingClassifier(baseleaner, n_estimators=bagging_number)  # , n_jobs = -1)

                        clf.fit(X_Training, Y_Training)
                        Output = clf.predict(X_Testing)

                        ac_positive1 = 0
                        ac_positive2 = 0
                        ac_positive3 = 0

                        ac_negative = 0
                        """
                        for tab in range(len(Output)):
                            if Output[tab]==positive_sign and Output[tab]==int(Y_Testing[tab]):
                                ac_positive += 1
                            if Output[tab]==negative_sign and Output[tab]==int(Y_Testing[tab]):
                                ac_negative += 1
                        """
                        for tab in range(len(Output)):
                            if Output[tab] == 2 and Output[tab]==int(Y_Testing[tab]):
                                ac_positive1 += 1
                            if Output[tab] == 3 and Output[tab]==int(Y_Testing[tab]):
                                ac_positive2 += 1
                            if Output[tab] == 4 and Output[tab]==int(Y_Testing[tab]):
                                ac_positive3 += 1
                            if Output[tab] <=1 and Output[tab]==int(Y_Testing[tab]):
                                ac_negative += 1

                        recall_positive_len1 = 0
                        recall_positive_len2 = 0
                        recall_positive_len3 = 0
                        recall_negative_len = 0
                        for each in Y_Testing:
                            if each == 2:
                                recall_positive_len1 += 1
                            if each == 3:
                                recall_positive_len2 += 1
                            if each == 4:
                                recall_positive_len3 += 1
                            if each == 1:
                                recall_negative_len += 1

                        precision_positive_len1 = 0
                        precision_positive_len2 = 0
                        precision_positive_len3 = 0
                        precision_negative_len = 0
                        for each in Output:
                            if int(each) == 2:
                                precision_positive_len1 += 1
                            if int(each) == 3:
                                precision_positive_len2 += 1
                            if int(each) == 4:
                                precision_positive_len3 += 1
                            if int(each) == 1:
                                precision_negative_len += 1
                        #g_mean=np.sqrt(float(ac_positive*ac_negative)/CC*(len(Y_Testing)-CC))

                        #g_mean=np.sqrt(float(ac_positive*ac_negative)/(count_positive*count_negative))
                        #auc=float(get_auc(np.array(Output),Y_Testing,positive_sign))
                        #auc=float(roc_auc_score(Y_Testing, np.array(Output)))
                        #print("TP is..."+str(float(ac_positive)/Output.count(positive_sign)))
                        #print("TN is..."+str(float(ac_negative)/Output.count(negative_sign)))
                        #ACC = float(ac_positive+ac_negative)/len(Output)


                        print("Revall of :"+eachMethod)
                        print("Anomaly1:")
                        print(Y_Testing)
                        print(ac_positive1/(1.0*recall_positive_len1))
                        print("Anomaly2:")
                        print(ac_positive2/(1.0*recall_positive_len2))
                        print("Anomaly3:")
                        print(ac_positive3/(1.0*recall_positive_len3))
                        print("Negative:")
                        print(ac_negative/(1.0*recall_negative_len))
                        print("---------------------------------")
                        print("Precesion of :"+eachMethod)
                        print("Anomaly1:")
                        print(ac_positive1/(1.0*precision_positive_len1))
                        print("Anomaly2:")
                        print(ac_positive2/(1.0*precision_positive_len2))
                        print("Anomaly3:")
                        print(ac_positive3/(1.0*precision_positive_len3))
                        print("Negative:")
                        print(ac_negative/(1.0*precision_negative_len))
                        #cross_folder_acc_list.append(ACC*100)
                        #cross_folder_g_mean_list.append(g_mean*100)
                        #cross_folder_auc_list.append(auc*100)


                    for tab1 in range(int(len(cross_folder_auc_list)/cross_folder)):
                        temp=0.0
                        for tab2 in range(cross_folder):
                            temp += cross_folder_auc_list[tab1*cross_folder+tab2]
                        temp=temp/float(cross_folder)
                        Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(temp)


                    for tab1 in range(int(len(cross_folder_acc_list)/cross_folder)):
                        temp=0.0
                        for tab2 in range(cross_folder):
                            temp += cross_folder_acc_list[tab1*cross_folder+tab2]
                        temp=temp/float(cross_folder)
                        Temp_SubFeature_ACC_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(temp)

                    for tab1 in range(int(len(cross_folder_g_mean_list)/cross_folder)):
                        temp=0.0
                        for tab2 in range(cross_folder):
                            temp += cross_folder_g_mean_list[tab1*cross_folder+tab2]
                        temp=temp/float(cross_folder)
                        Temp_SubFeature_G_mean_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(temp)

                deviation_auc=0.0
                #mean_auc=Compute_average_list(Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)])
                for tab in range(len(Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)])):
                    temp = Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)][tab]
                    #deviation_auc=deviation_auc+((temp-mean_auc)*(temp-mean_auc))
                #deviation_auc/=Iterations

                Deviation_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(deviation_auc)


                #Temp_Bagging_Auc_list[eachMethod+"_BN_"+str(bagging_number)].append(Compute_average_list(Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)]))
                #Temp_Bagging_G_mean_list[eachMethod+"_BN_"+str(bagging_number)].append(Compute_average_list(Temp_SubFeature_G_mean_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)]))
                #Temp_Bagging_ACC_list[eachMethod+"_BN_"+str(bagging_number)].append(Compute_average_list(Temp_SubFeature_ACC_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)]))

            #The_Top_K=Top_K_List[Temp_Bagging_Auc_list.index(max(Temp_Bagging_Auc_list))]
            #print(Temp_Bagging_Auc_list)
            #print(The_Top_K)

            #Auc_list[eachMethod].append(Compute_average_list(Temp_Bagging_Auc_list[eachMethod+"_BN_"+str(bagging_number)]))
            #G_mean_list[eachMethod].append(Compute_average_list(Temp_Bagging_G_mean_list[eachMethod+"_BN_"+str(bagging_number)]))
            #ACC_list[eachMethod].append(Compute_average_list(Temp_Bagging_ACC_list[eachMethod+"_BN_"+str(bagging_number)]))

        #print(Auc_list)
        #print(ACC_list)
        #print(G_mean_list)
        #print(Temp_Bagging_Auc_list)
        #print(Temp_SubFeature_Auc_list)

        #print("auclist.....for......"+str()+"---------MaxAUC:"+str(max(plot_auc_list))+"---------MeanAUC:"+str(sum(plot_auc_list)/float(len(plot_auc_list)))+"-----Deviation:"+str(deviation_auc))
        #print("gmeanlist.....for......"+str()+"---------MaxGmean:"+str(max(plot_g_mean_list))+"---------MeanGmean:"+str(sum(plot_g_mean_list)/float(len(plot_g_mean_list))))
    with open(os.path.join(out_put_path,filename+"Info_G_mean_List.txt"),"w")as fout:
        for eachk,eachv in G_mean_list.items():
            fout.write(eachk)
            fout.write(":\t\t")
            for each in eachv:
                fout.write("%.3f"%(each))
                fout.write("\t,")
            fout.write('\n')
    with open(os.path.join(out_put_path,filename+"Info_Bagging_G_mean_List.txt"),"w")as fout:
        for eachk,eachv in Temp_Bagging_G_mean_list.items():
            fout.write(eachk)
            fout.write(":\t\t")
            for each in eachv:
                fout.write("%.3f"%(each))
                fout.write("\t,")
            fout.write('\n')
    with open(os.path.join(out_put_path,filename+"Info_SubFeature_G_mean_List.txt"),"w")as fout:
        for eachk,eachv in Temp_SubFeature_G_mean_list.items():
            fout.write(eachk)
            fout.write(":\t\t")
            for each in eachv:
                fout.write("%.3f"%(each))
                fout.write("\t,")
            #print(Deviation_list)
            fout.write(str(Deviation_list[eachk]))
            fout.write('\n')
    with open(os.path.join(out_put_path,filename+"Info_ACC_List.txt"),"w")as fout:
        for eachk,eachv in ACC_list.items():
            fout.write(eachk)
            fout.write(":\t\t")
            for each in eachv:
                fout.write("%.3f"%(each))
                fout.write("\t,")
            fout.write('\n')
    with open(os.path.join(out_put_path,filename+"Info_Bagging_ACC_List.txt"),"w")as fout:
        for eachk,eachv in Temp_Bagging_ACC_list.items():
            fout.write(eachk)
            fout.write(":\t\t")
            for each in eachv:
                fout.write("%.3f"%(each))
                fout.write("\t,")
            fout.write('\n')
    with open(os.path.join(out_put_path,filename+"Info_SubFeature_ACC_List.txt"),"w")as fout:
        for eachk,eachv in Temp_SubFeature_ACC_list.items():
            fout.write(eachk)
            fout.write(":\t\t")
            for each in eachv:
                fout.write("%.3f"%(each))
                fout.write("\t,")
            #print(Deviation_list)
            fout.write(str(Deviation_list[eachk]))
            fout.write('\n')


    with open(os.path.join(out_put_path,filename+"Info_Auc_List.txt"),"w")as fout:
        for eachk,eachv in Auc_list.items():
            fout.write(eachk)
            fout.write(":\t\t")
            for each in eachv:
                fout.write("%.3f"%(each))
                fout.write("\t,")
            fout.write('\n')
    with open(os.path.join(out_put_path,filename+"Info_Bagging_Auc_List.txt"),"w")as fout:
        for eachk,eachv in Temp_Bagging_Auc_list.items():
            fout.write(eachk)
            fout.write(":\t\t")
            for each in eachv:
                fout.write("%.3f"%(each))
                fout.write("\t,")
            fout.write('\n')
    with open(os.path.join(out_put_path,filename+"Info_SubFeature_Auc_List.txt"),"w")as fout:
        for eachk,eachv in Temp_SubFeature_Auc_list.items():
            fout.write(eachk)
            fout.write(":\t\t")
            for each in eachv:
                fout.write("%.3f"%(each))
                fout.write("\t,")
            #print(Deviation_list)
            fout.write(str(Deviation_list[eachk]))
            fout.write('\n')

if __name__=='__main__':
    global positive_sign,negative_sign,count_positive,count_negative,out_put_path
    #os.chdir("/home/grads/mcheng223/IGBB")
    positive_sign=1
    negative_sign=0
    count_positive=0
    count_negative=0
    input_data_path = os.path.join(os.getcwd(),"BGPData")

    out_put_path = os.path.join(os.getcwd(),"Output_BGPData")
    if not os.path.isdir(out_put_path):
        os.makedirs(out_put_path)
    filenamelist=os.listdir(input_data_path)

    Method_Dict={"IGBB":1}
    #Method_Dict={"IGBB":1,"DT":2,"SVM":3,"KNN":5,"IGBB0":6}
    for eachfile in filenamelist:
        if 'HB' in eachfile  and 'Multi' in eachfile and '.txt' in eachfile:
            pass
        else:
            continue
        Main(Method_Dict,eachfile)

    print(time.time()-start)

"""

{'KNN': [68.79363135333742], 'SVM': [70.69197795468463], 'IGBB': [72.44335578689528], 'LR': [46.87078995713411], 'IGBB0': [71.96570728720147], 'DT': [71.87997550520514]}
206.680423021

"""