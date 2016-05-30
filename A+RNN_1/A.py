import os
from shutil import copyfile
inputpath = "/home/cheng/Dropbox/BGP_ANOMALY"
filelist = os.listdir(inputpath)
outputpath = inputpath.replace("BGP_ANOMALY","BGP_ANOMALY2")

def G_(inputpath,outputpath):
    if not os.path.isdir(outputpath):
        os.makedirs(outputpath)
    for each in filelist:
        if os.path.isdir(os.path.join(inputpath,each)):
            inputpath1 = os.path.join(outputpath,each)
            outputpath1 = inputpath1.replace("BGP_ANOMALY","BGP_ANOMALY2")

            try:
                os.makedirs(outputpath1)

            except:
                try:
                    G_(inputpath1,outputpath1)
                except:
                    pass
        elif not ('.txt' in each or '.zip' in each):
            copyfile(os.path.join(inputpath,each), os.path.join(outputpath,each))
G_(inputpath,outputpath)

