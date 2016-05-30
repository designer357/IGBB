'''Trains a LSTM on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF+LogReg.

Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import sys
import os
from sklearn import svm,datasets,preprocessing,linear_model
global positive_sign,negative_sign,count_positive,count_negative,out_put_path,modified_positive,modified_negative
positive_sign=-1
negative_sign=1
modified_positive = 0
modified_negative = 1
count_positive=0
count_negative=0
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
def LoadData(input_data_path,filename):
    """
    global input_data_path,out_put_path
modified_negative
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
            temp.append(int(v))
        Data[tab].extend(temp)
        Data[tab].append(int(y_svmformat[tab]))
    Data=np.array(Data)
    np.random.shuffle(Data)
    np.random.shuffle(Data)
    return Data
    """
    global positive_sign,negative_sign,count_positive,count_negative,out_put_path,modified_negative,modified_positivei
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
        elif filename == 'BGP_DATA.txt':
            negative_flag = '1.0'
    #with open("AS_Filtering_Error_AS_286_half_minutes.txt") as fin:
        Data=[]

        for each in fin:
            if '@' in each:
                continue
            val=each.split(",")
            if len(val)>0 or val[-1].strip()=="negative" or val[-1].strip()=="positive":
                #print(each)
                if val[-1].strip()== negative_flag:
                    val[-1]=modified_negative
                    count_negative += 1
                else:
                    val[-1]= modified_positive
                    count_positive += 1
                try:
                    val=map(lambda a:float(a),val)
                except:
                    val=map(lambda a:str(a),val)

                val[-1]=int(val[-1])
                Data.append(val)
        Data=np.array(Data)
        return Data


def reConstruction(window_size,data,label):
    newdata = []
    newlabel = []
    L = len(data)
    D = len(data[0])
    #W = 20
    interval = 2
    index = 0
    newdata_count = 0
    initial_value = -999
    while index+window_size < L:
        newdata.append(initial_value)
        newlabel.append(initial_value)
        Sequence = []
        for i in range(window_size):
            Sequence.append(data[index+i])
            newlabel[newdata_count] = label[index+i]
        index += interval
        newdata[newdata_count]=Sequence
        newdata_count += 1
    return np.array(newdata),np.array(newlabel)


def Main(window_size,lstm_size):
    input_data_path = os.path.join(os.getcwd(),"Data4")
    filenamelist=os.listdir(input_data_path)
    for eachfile in filenamelist:
        Data_=LoadData(input_data_path,eachfile)
        Positive_Data=Data_[Data_[:,-1]==modified_positive]
        Negative_Data=Data_[Data_[:,-1]==negative_sign]
        print("IR is :"+str(float(len(Negative_Data))/len(Positive_Data)))
        cross_folder=1
        for tab_cross in range(cross_folder):
            Training_Data=np.concatenate((Positive_Data[:int(len(Positive_Data)*0.5),:],Negative_Data[:int(len(Positive_Data)*0.5),:]))
            Testing_Data=np.concatenate((Positive_Data[int(len(Positive_Data)*0.5):,:],Negative_Data[int(len(Positive_Data)*0.5):,:]))

            scaler = preprocessing.StandardScaler()
            batch_size = 200

            (X_train,y_train) = reConstruction(window_size,scaler.fit_transform(Training_Data[:,:-1]),Training_Data[:,-1])
            (X_test,y_test) = reConstruction(window_size,scaler.fit_transform(Testing_Data[:,:-1]),Testing_Data[:,-1])
            lstm_object = LSTM(lstm_size,input_length=window_size,input_dim=33)
            print('Build model...'+'Window Size is '+str(window_size)+' LSTM Size is '+str(lstm_size))
            model = Sequential()
            model.add(lstm_object)#X.shape is (samples, timesteps, dimension)
            model.add(Dense(output_dim=1))
            model.add(Activation("sigmoid"))
            model.compile( optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
            model.fit(X_train, y_train, batch_size=batch_size,nb_epoch=10)
            score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
            print('Test score2:', score)
            print('Test accuracy2:', acc)
            return acc

if __name__=="__main__":

    total_args = list(sys.argv)
    print("Start->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->")
    total_args.pop(0)
    window_size = int(total_args[0])
    lstm_size = int(total_args[1])
    acc = Main(window_size,lstm_size)
    with open("RESULT.txt","a")as fout:
        fout.write("Window Size: "+str(window_size)+" LSTM Size: "+str(lstm_size)+'\t\tAccuracy: '+str(acc))
        fout.write('\n')

