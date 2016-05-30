import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


with open("RESULT_Nimda_Slammer.txt.txt")as fin:
    lines = fin.readlines()
    window_size_list = []
    lstm_size_list = []
    acc_list = []
    for eachline in lines:
        try:
            window_size,lstm_size,acc = eachline.replace(' LSTM','\t\tLSTM ').split('\t\t',2)
            window_size_list.append(int(window_size.strip().split(':')[-1].strip()))
            lstm_size_list.append(int(lstm_size.strip().split(':')[-1].strip()))
            acc_list.append(float(acc.strip().split(':')[-1].strip()))
        except:
            pass
    max_index = acc_list.index(max(acc_list))
    print("The max acc is "+str(acc_list[max_index])+" and the window size is "+str(window_size_list[max_index])+" the lstm size is "+str(lstm_size_list[max_index]))

X = np.array(window_size_list)
Y = np.array(lstm_size_list)
Z = np.array(acc_list)


from scipy.interpolate import griddata

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
# note this: you can skip rows!

xi = np.linspace(X.min(),X.max(),100)
yi = np.linspace(Y.min(),Y.max(),100)
# VERY IMPORTANT, to tell matplotlib how is your data organized
zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='cubic')

CS = plt.contour(xi,yi,zi,15,linewidths=0.5,color='k')
ax = fig.add_subplot(1, 2, 2, projection='3d')

xig, yig = np.meshgrid(xi, yi)

surf = ax.plot_surface(xig, yig, zi,linewidth=0)

plt.show()

print(Z)
print(X)
