from sklearn import *
import numpy as np
from matplotlib import *
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import datasets
def plot_decision_boundry(pred_func):
    #Set min and max values and give it some padding
    x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
    y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
    h = 0.01

    #Generate a grid of points with distance h between them
    xx , yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

    #Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)

    #Plot the contour and training samples
    plt.contourf(xx,yy,Z)
    plt.scatter(X[:,0],X[:,1],c = y)

#Generate a data set and plot it
np.random.seed(0)
X,y = datasets.make_moons(200,noise = 0.20 )
print(X.shape)
plt.scatter(X[:,0],X[:,1],s = 40,c = y)
#plt.show()

# Train our Logistic Regression Classifier
clf = linear_model.LogisticRegressionCV()
clf.fit(X,y)

#Plot the decision boundry
plot_decision_boundry(lambda x: clf.predict(x))
plt.title("Logistic Regression")
plt.show()

