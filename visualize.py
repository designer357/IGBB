import matplotlib.pyplot as plt
import matplotlib
"""==================== Visualize ======================"""
def set_style():
    plt.style.use(['seaborn-paper'])
    matplotlib.rc("font", family="serif")
set_style()
def plot_error_rate(er_train, er_test):
    global max_n,interval_n
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