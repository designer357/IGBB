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


#max_features = 128
#maxlen = 33  # cut texts after this number of words (among top max_features most common words)
batch_size = 200
lstm_size = 16
def reConstruction(data,label):
    newdata = []
    newlabel = []
    L = len(data)
    D = len(data[0])
    W = 20
    interval = 2
    index = 0
    newdata_count = 0
    initial_value = -999
    while index+W < L:
        newdata.append(initial_value)
        newlabel.append(initial_value)
        Sequence = []
        for i in range(W):
            Sequence.append(data[index+i])
            newlabel[newdata_count] = label[index+i]
        index += interval
        newdata[newdata_count]=Sequence
        newdata_count += 1
    return np.array(newdata),np.array(newlabel)
print('Loading data...')
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                     #test_split=0.2)

input_data_path = os.path.join(os.getcwd(),"Data4")
filenamelist=os.listdir(input_data_path)
for eachfile in filenamelist:
    Data_=LoadData(input_data_path,eachfile)
    #print("IR is :"+str(float(count_negative)/count_positive))

    Positive_Data=Data_[Data_[:,-1]==modified_positive]
    Negative_Data=Data_[Data_[:,-1]==negative_sign]
    print("IR is :"+str(float(len(Negative_Data))/len(Positive_Data)))
    count_positive = len(Positive_Data)
    count_negative = len(Negative_Data)
    cross_folder=1
    Positive_Data_Index_list=[i for i in range(len(Positive_Data))]
    Negative_Data_Index_list=[i for i in range(len(Negative_Data))]
    for tab_cross in range(cross_folder):
        Positive_Data_Index_Training=[]
        Positive_Data_Index_Testing=[]
        Negative_Data_Index_Training=[]
        Negative_Data_Index_Testing=[]

        #for tab_positive in Positive_Data_Index_list:
            #if int((cross_folder-tab_cross-1)*len(Positive_Data)/cross_folder)<tab_positive<int((cross_folder-tab_cross)*len(Positive_Data)/cross_folder):
                #Positive_Data_Index_Testing.append(tab_positive)
            #else:
                #Positive_Data_Index_Training.append(tab_positive)
        #for tab_negative in Negative_Data_Index_list:
            #if int((cross_folder-tab_cross-1)*len(Negative_Data)/cross_folder)<tab_negative<int((cross_folder-tab_cross)*len(Negative_Data)/cross_folder):
                #Negative_Data_Index_Testing.append(tab_negative)
            #else:
                #Negative_Data_Index_Training.append(tab_negative)
        #print("111111111111111111111111111111")
        #print(Positive_Data_Index_Training)
        #print("222222222222222222222222222222")
        """
        Positive_Training_Data=np.array(Positive_Data)[Positive_Data_Index_Training]
        Positive_Testing_Data=np.array(Positive_Data)[Positive_Data_Index_Testing]
        Negative_Training_Data=np.array(Negative_Data)[Negative_Data_Index_Training]
        Negative_Testing_Data=np.array(Negative_Data)[Negative_Data_Index_Testing]
        Training_Data=np.concatenate((Positive_Training_Data,Negative_Training_Data))
        """
        Training_Data=np.concatenate((Positive_Data[:int(len(Positive_Data)*0.5),:],Negative_Data[:int(len(Positive_Data)*0.5),:]))
        Testing_Data=np.concatenate((Positive_Data[int(len(Positive_Data)*0.5):,:],Negative_Data[int(len(Positive_Data)*0.5):,:]))

        scaler = preprocessing.StandardScaler()
        #X_train = scaler.fit_transform(X_train)
        #X_test = scaler.fit_transform(X_test)






        #Testing_Data=np.append(Positive_Testing_Data,Negative_Testing_Data,axis=0)
        #X_test = Testing_Data[:,:-1]
        #y_test=Testing_Data[:,-1]

        output_list= []
        model_list = []
        for lstm_size in range(30,31,10):
            (X_train,y_train) = reConstruction(scaler.fit_transform(Training_Data[:,:-1]),Training_Data[:,-1])
            (X_test,y_test) = reConstruction(scaler.fit_transform(Testing_Data[:,:-1]),Testing_Data[:,-1])
            print("lstm_size is --------------------------------------------------------------------------"+str(lstm_size))
            lstm_object1 = LSTM(20,input_length=20,input_dim=33)
            lstm_object2 = LSTM(20,input_length=20,input_dim=33)

            print('Build model...')

            model1 = Sequential()

            #model_list.append(Sequential())
            # try using different optimizers and different optimizer configs
            #model.compile(loss='hinge',optimizer='rmsprop',metrics=['accuracy'])
            #model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

            print('Loading data...')

            #print((X_train[0]))
            #model.add(Embedding(60000, 128, input_length=33, dropout=0.2))
            #model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))

            print('Build model...')
            #actionmethod = "relu"
            #model.add(Embedding(max_features, 64, input_length=maxlen))
            #model.add(Dense(500, input_dim=33, init='uniform'))
            #model.add(Dropout(0.15))
            model1.add(lstm_object1)#X.shape is (samples, timesteps, dimension)

            #40Test score: 0.53953447938Test accuracy: 0.776129040026

            #model.add(Dense(500, input_dim=33, init='uniform'))
            #model.add(Dropout(0.15))
            #model.add(Dense(output_dim=4, input_dim=20))
            #model.add(Activation(actionmethod))
            #model.add(Dropout(0.05))
            #model.add(Dense(output_dim=20, input_dim=100))
            #model.add(Activation(actionmethod))
            #model.add(Dropout(0.15))

            #model.add(Embedding(1000, lstm_size, input_length=50))
            #model.add(LSTM(32, input_shape=(10, 50)))

            model1.add(Dense(output_dim=1))
            model1.add(Activation("sigmoid"))
            #model.compile(loss='binary_crossentropy', optimizer='sgd', class_mode='binary')
            model1.compile( optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
            #model.fit(X_train, y_train, nb_epoch=15, batch_size=32)  # starts training
            #model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10, validation_data=(X_test, y_test))
            model1.fit(X_train, y_train, batch_size=batch_size,nb_epoch=10)

            #model.fit(X_train,y_train)
            #score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
            #print('Test score:', score)
            #print('Test accuracy:', acc)
            #model.fit(X_train, y_train, nb_epoch=80, batch_size=20)
            #model.fit(X_train,y_train)
            #print(model.evaluate(X_test, y_test, show_accuracy=True))
            #print(model.predict_classes(X_test))
            score1, acc1 = model1.evaluate(X_test, y_test, batch_size=batch_size)
            print('Test score:', score1)
            print('Test accuracy:', acc1)


            model2 = Sequential()
            model2.add(lstm_object2)#X.shape is (samples, timesteps, dimension)
            model2.add(Dense(output_dim=1))
            model2.add(Activation("sigmoid"))
            #model.compile(loss='binary_crossentropy', optimizer='sgd', class_mode='binary')
            model2.compile( optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
            #model.fit(X_train, y_train, nb_epoch=15, batch_size=32)  # starts training
            #model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10, validation_data=(X_test, y_test))
            model2.fit(X_train, y_train, batch_size=batch_size,nb_epoch=10)

            #model.fit(X_train,y_train)
            #score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
            #print('Test score:', score)
            #print('Test accuracy:', acc)
            #model.fit(X_train, y_train, nb_epoch=80, batch_size=20)
            #model.fit(X_train,y_train)
            #print(model.evaluate(X_test, y_test, show_accuracy=True))
            #print(model.predict_classes(X_test))
            score2, acc2 = model2.evaluate(X_test, y_test, batch_size=batch_size)
            print('Test score:', score2)
            print('Test accuracy:', acc2)



            #output_list.append(acc1)
            output_list.append(acc2)

            #classes = model.predict_classes(X_test, batch_size=32)
            #print("True Label:\n")
            #print(y_test)
            #print("Predict Label:\n")
print(output_list)
print(model_list)


