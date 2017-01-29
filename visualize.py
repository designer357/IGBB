import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn
"""==================== Visualize ======================"""
def set_style():
    plt.style.use(['seaborn-paper'])
    matplotlib.rc("font", family="serif")
#set_style()
seaborn.set_context('paper')
def plotting(flag,filename,method_dict,bagging_list,results,text=''):
    color_list = ['r','y', 'g','#FF8C00','#FD8CD0','c', 'b','m']

    for tab in range(len(method_dict)):
        plt.plot(bagging_list,results[:,tab],color_list[tab],label={v:k for k, v in method_dict.items()}[tab])
    plt.legend()
    plt.tick_params(labelsize=12)
    plt.xlabel('Bagging Size',fontsize=12)
    plt.ylabel(flag,fontsize=12)
    plt.ylim(0,1.1)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.join(os.getcwd(),'images'),filename + text +'.png'),dpi=400)
    plt.savefig(os.path.join(os.path.join(os.getcwd(),'images'),filename + text +'.pdf'),dpi=400)
    plt.clf()

def plot_error_rate(max_n,interval_n,er_train, er_test):
    x = [i for i in range(10, max_n, interval_n)]
    plt.plot(x,er_train,'b',label='Ada.Boost')
    plt.plot(x,er_test,'r',label='IGBoosting')
    plt.tick_params(labelsize=12)
    plt.xlabel('Number of iterations',fontsize=12)
    plt.ylabel('Error rate', fontsize=12)
    plt.ylim(0.155,0.19)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig("IGBB_VS_ADABOOST.png",dpi=400)
    plt.show()