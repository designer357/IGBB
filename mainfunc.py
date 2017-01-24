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
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA,KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm,datasets,preprocessing,linear_model
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier



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
def top_feat(data,label,w,k):
    new_data = np.multiply(data.T,w).T
    y_=label
    top_list=[]
    for tab in range(len(data[0])):
        top_list.append(informationGain(new_data[:, tab], y_))
        #if len(data_copy[:,tab])==len(w):
            #for i in range(len(data_copy[:,tab])):
                #data_copy[:,tab][i]=w[i]*data_copy[:,tab][i]
        #else:
            #print("Error! Data_[column] not equal to weight!")
        #top_list.append(informationGain(data_copy[:,tab],y_))
    result=(sorted(enumerate(top_list),key=lambda a:a[1],reverse=True))
    label_=[e[0] for e in result]
    return label_[:k]


# Training
def train(data,label,Top_K,numIt = 1000,flag = 0):
    SubSpace_WeakClassifiers={"weakClassifiers":[],"subSpace":[]}
    #weakClassifiers = []


    Sub_Features=sorted(top_feat(data,label,Top_K))

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
        Sub_Features=sorted(top_feat(data,label,D,Top_K))
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



def igbb(bagging_size,boosting_iter):
    clf = AdaBoostClassifier()
    for bagging_number in range(bagging_size):
        for iter in range(boosting_iter):



#def InformationGainBoosting(Iterations):
def Main(Method_Dict,bagging_size,input_data_path,out_put_path,filename):
    #Name_Str_List = ["Code_Red_I_NimdaSlammer.txt","Code_Red_I_SlammerNimda.txt","Nimda_SlammerCode_Red_I.txt"]
    print(filename+" is processing......")
    data = loaddata.loadData(input_data_path, filename)


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

    for bagging_number in range(bagging_size):
        print("The Bagging Number is " + str(bagging_number) + "...")

        X_train, Y_train, X_test, Y_test = loaddata.cross_tab(data, 2, 1)


        for eachMethod,methodLabel in Method_Dict.items():
            print(eachMethod+" is running...")
            Auc_list[eachMethod] = []
            ACC_list[eachMethod] = []
            G_mean_list[eachMethod] = []
            Top_K_List = []
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
                        #import matplotlib.pyplot as plt
                        #plt.plot(Negative_Data_Index_Testing,Negative_Testing_Data,'g')
                        #plt.plot(Positive_Data_Index_Testing,Positive_Testing_Data,'r')
                        #plt.show()

                        #Features=[i for i in range(len(Positive_Training_Data[0])-1)]
                        #Sub_Features=Features[:Top_K]
                        N = len(Negative_Testing_Data)
                        Negative_Testing_Data2 = Negative_Testing_Data[N-len(Positive_Testing_Data)-1:N-1,:]
                        Testing_Data=np.append(Positive_Testing_Data,Negative_Testing_Data2,axis=0)

                        #for each_str in Name_Str_List:
                            #try:
                                #print(filename)
                                #print(each_str)
                                #print(each_str.strip().replace(filename.replace(".txt","").strip(),""))
                                #Testing_Data = LoadData(input_data_path2,each_str.replace(filename.replace(".txt",""),""))
                            #except:
                                #continue
                        Y_Testing=Testing_Data[:,-1]

                        ac_positive=0
                        ac_negative=0
                        if bagging_number==1:
                            #Training_Data=np.concatenate((Positive_Training_Data,Negative_Training_Data))
                            Training_Data = Data_
                            #X_Training = Training_Data[:,Sub_Features]
                            Y_Training = Training_Data[:,-1]

                            #D=[1/float(len(Y_Training)) for i in range(len(Y_Training))]
                            #Sub_Features=sorted(top_feat(Training_Data[:,:-1],Y_Training,D,Top_K))
                            #Sub_Features = [1,3,4,5]
                            #print(Training_Data)

                            X_Training = Training_Data[:,:-1]
                            X_Testing=Testing_Data[:,:-1]

                            if methodLabel==1:
                                #clf = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.1,max_depth=2)
                                clf = AdaBoostClassifier()
                                #classifier = train(X_Training,Y_Training,Top_K)
                                #TempList=test(X_test,classifier)
                            elif methodLabel==2:
                                clf=tree.DecisionTreeClassifier()
                            elif methodLabel==3:
                                scaler = preprocessing.StandardScaler()
                                X_Training = scaler.fit_transform(X_Training)
                                X_Testing = scaler.fit_transform(X_Testing)
                                clf = svm.SVC(kernel="rbf", gamma=0.001)
                            elif methodLabel==4:
                                clf = linear_model.LogisticRegression()
                            elif methodLabel==5:
                                clf = KNeighborsClassifier(5)

                            #clf = AdaBoostClassifier()
                            #classifier = train(X_Training,Y_Training,Top_K,100)
                            #result=test(X_Testing,classifier)

                            clf.fit(X_Training,Y_Training)
                            result=clf.predict(X_Testing)

                            Output=[]
                            if len(result)==len(Y_Testing):
                                for tab in range(len(Y_Testing)):
                                    Output.append(int(result[tab]))
                            else:
                                print("Error!")

                        else:
                            VotingList=[[] for i in range(bagging_number)]
                            for t in range(bagging_number):
                                #Positive_Data_Samples=RANDOM.sample(Positive_Training_Data,int(len(Positive_Training_Data)))
                                Positive_Data_Samples = Positive_Training_Data
                                Negative_Data_Samples=RANDOM.sample(Negative_Training_Data,len(Positive_Data_Samples))

                                TrainingSamples=np.concatenate((Negative_Data_Samples,Positive_Data_Samples))
                                #X_Training=TrainingSamples[:,Sub_Features]
                                Y_Training=TrainingSamples[:,-1]

                                #D=[1/float(len(Y_Training)) for i in range(len(Y_Training))]
                                #Sub_Features=sorted(top_feat(TrainingSamples[:,:-1],Y_Training,D,Top_K))
                                #print("Bagging : "+str(t+1))
                                #print(Sub_Features)
                                X_Training=TrainingSamples[:,:-1]
                                X_Testing=Testing_Data[:,:-1]

                                #scaler = preprocessing.MinMaxScaler()
                                #X_Training = scaler.fit_transform(X_Training)
                                #X_Testing = scaler.fit_transform(X_Testing)
                                if methodLabel==1:
                                    #clf = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.1,max_depth=2)
                                    clf = AdaBoostClassifier()
                                    #clf = train(X_Training,Y_Training,Top_K)
                                    #TempList=test(X_test,classifier)
                                elif methodLabel==2:
                                    clf=tree.DecisionTreeClassifier()
                                elif methodLabel==3:

                                    clf = svm.SVC(kernel="rbf", gamma=0.001,C=1000)
                                elif methodLabel==4:
                                    clf = linear_model.LogisticRegression()
                                elif methodLabel==5:
                                    clf = KNeighborsClassifier(3)
                                #print("THE METHOD IS "+str(methodLabel))

                                if methodLabel==1:
                                    D = [1 / float(len(Y_Training)) for i in range(len(Y_Training))]
                                    Sub_Features=sorted(top_feat(TrainingSamples[:,:-1],Y_Training,D,10))
                                    X_Training=X_Training[:,Sub_Features]

                                if methodLabel==3 or methodLabel==4 or methodLabel==5:
                                    scaler = preprocessing.MinMaxScaler()
                                    X_Training = scaler.fit_transform(X_Training)
                                    X_Testing = scaler.fit_transform(X_Testing)

                                clf.fit(X_Training, Y_Training)

                                TempList = clf.predict(X_Testing)

                                VotingList[t].extend(TempList)

                            TempOutput=[[] for i in range(len(VotingList[0]))]
                            Output=[]
                            for tab_i in range(len(VotingList[0])):
                                for tab_j in range(len(VotingList)):
                                    TempOutput[tab_i].append(VotingList[tab_j][tab_i])
                            for tab_i in range(len(TempOutput)):
                                if TempOutput[tab_i].count(positive_sign)>TempOutput[tab_i].count(negative_sign):
                                    Output.append(positive_sign)
                                else:
                                    Output.append(negative_sign)

                        for tab in range(len(Output)):
                            if Output[tab]==positive_sign and Output[tab]==int(Y_Testing[tab]):
                                ac_positive += 1
                            if Output[tab]==negative_sign and Output[tab]==int(Y_Testing[tab]):
                                ac_negative += 1
                        g_mean=np.sqrt(float(ac_positive*ac_negative)/(list(Y_Testing).count(positive_sign)*list(Y_Testing).count(negative_sign)))
                        #auc=float(get_auc(np.array(Output),Y_Testing,positive_sign))
                        auc=float(roc_auc_score(Y_Testing, np.array(Output)))
                        #print("TP is..."+str(float(ac_positive)/Output.count(positive_sign)))
                        #print("TN is..."+str(float(ac_negative)/Output.count(negative_sign)))
                        ACC = float(ac_positive+ac_negative)/len(Output)

                        cross_folder_acc_list.append(ACC*100)
                        cross_folder_g_mean_list.append(g_mean*100)
                        cross_folder_auc_list.append(auc*100)


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
                mean_auc=evaluation.average_list(Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)])
                for tab in range(len(Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)])):
                    temp = Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)][tab]
                    deviation_auc=deviation_auc+((temp-mean_auc)*(temp-mean_auc))
                deviation_auc/=Iterations

                Deviation_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)].append(deviation_auc)


                Temp_Bagging_Auc_list[eachMethod+"_BN_"+str(bagging_number)].append(evaluation.average_list(Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)]))
                Temp_Bagging_G_mean_list[eachMethod+"_BN_"+str(bagging_number)].append(evaluation.average_list(Temp_SubFeature_G_mean_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)]))
                Temp_Bagging_ACC_list[eachMethod+"_BN_"+str(bagging_number)].append(evaluation.average_list(Temp_SubFeature_ACC_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)]))

            #The_Top_K=Top_K_List[Temp_Bagging_Auc_list.index(max(Temp_Bagging_Auc_list))]
            #print(Temp_Bagging_Auc_list)
            #print(The_Top_K)

            Auc_list[eachMethod].append(evaluation.average_list(Temp_Bagging_Auc_list[eachMethod+"_BN_"+str(bagging_number)]))
            G_mean_list[eachMethod].append(evaluation.average_list(Temp_Bagging_G_mean_list[eachMethod+"_BN_"+str(bagging_number)]))
            ACC_list[eachMethod].append(evaluation.average_list(Temp_Bagging_ACC_list[eachMethod+"_BN_"+str(bagging_number)]))

        #print(Auc_list)
        print(ACC_list)
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
    global positive_sign,negative_sign
    #os.chdir("/home/grads/mcheng223/IGBB")
    positive_sign=-1
    negative_sign=1
    count_positive=0
    count_negative=0
    input_data_path = os.path.join(os.getcwd(),"BGPData")

    out_put_path = os.path.join(os.getcwd(),"Output_BGPData")
    if not os.path.isdir(out_put_path):
        os.makedirs(out_put_path)
    filenamelist=os.listdir(input_data_path)

    #Method_Dict={"DT":1,"LR":4}
    Method_Dict={"IGBB":1,"DT":2,"SVM":3,"LR":4,"KNN":5}
    for eachfile in filenamelist:
        if 'HB_Nimda' in eachfile and '.txt' in eachfile:
            if 'Multi' in eachfile:continue
            else:
                pass
        else:
            continue
        Main(Method_Dict,eachfile)

    print(time.time()-start)