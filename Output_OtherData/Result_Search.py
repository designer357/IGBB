import os
import re
import matplotlib.pyplot as plt
Method_Dict={"IGBB":1,"DT":2,"SVM":3,"LR":4,"KNN":5}
Output = {}
for eachk,eachv in Method_Dict.items():
    Output[eachk] = []

#print(os.listdir(os.getcwd()))
filelist = filter(lambda a:not "Reformed_" in a and "Bagging_Auc_List.txt" in a,os.listdir(os.getcwd()))
#print(filelist)

for eachfile in filelist:
    if os.path.isdir(os.path.join(os.getcwd(),eachfile.replace(".txt",""))):
        os.makedirs(os.path.join(os.getcwd(),eachfile.replace(".txt","")))
    with open(os.path.join(os.getcwd(),eachfile))as fin:
        val_list = fin.readlines()

    #plt.figure(20,20)
    figure_count = 0
    figure_pos_list=[(0,0),(0,2),(0,4),(1,1),(1,3)]
    for eachk,eachv in Method_Dict.items():
        bagging_list = []
        temp = []
        for eachline in val_list:
            if eachk in eachline:
                temp.append([])
                Output[eachk].append(eachline)
                bagging_list.append(int(eachline.split(':')[0].replace(eachk+"_BN_","")))
        templist = []
        bagging_list=sorted(bagging_list)
        #print(Output[eachk])
        for tab1 in range(len(bagging_list)):
            for tab2 in range(len(Output[eachk])):
                if Output[eachk][tab2].find(eachk+"_BN_"+str(bagging_list[tab1]))!=-1:
                    templist.append(Output[eachk][tab2])
        Output[eachk]=templist
        for tab in range(len(templist)):
            #print(templist[tab].strip().split(':')[1])
            temp[tab] = map(lambda a:float(a),filter(lambda a: len(a)>1,templist[tab].strip().split(':')[1].split(',')))
        feature_list = [(i+1)*5 for i in range(len(temp[0]))]
        #plt.subplot(2,3,figure_count)
        plt.subplot2grid(shape=(2,6),loc=figure_pos_list[figure_count],colspan=2)
        plt.title(eachk)
        plt.xlim(1,feature_list[-1]+1)
        plt.xlabel("Features")
        plt.ylim(70,100)
        plt.ylabel("G-mean")
        plt.plot(feature_list,temp[0],"rs-",label='B-'+str(eachk)+'_size_'+str(bagging_list[0]))
        plt.plot(feature_list,temp[1],"gs-",label='B-'+str(eachk)+'_size_'+str(bagging_list[1]))
        plt.plot(feature_list,temp[2],"bs-",label='B-'+str(eachk)+'_size_'+str(bagging_list[2]))
        plt.plot(feature_list,temp[3],"cs-",label='B-'+str(eachk)+'_size_'+str(bagging_list[3]))
        plt.plot(feature_list,temp[4],"ms-",label='B-'+str(eachk)+'_size_'+str(bagging_list[4]))
        #print((temp[0]))
        plt.grid()
        plt.tight_layout()

        legend = plt.legend(loc='best', shadow=True, fontsize='xx-small')
        legend.get_frame().set_facecolor('#00FFCC')
        figure_count += 1
        plt.savefig("haha.png")
        print("success")


    with open(os.path.join(os.getcwd(),"Reformed_"+eachfile),"w")as fout:
        for eachk,eachv in Output.items():
            fout.write(eachk+"\t:\n")
            for each in eachv:
                fout.write(each)


