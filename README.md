
## IGBB-Information Gain based Bagging Boosting classifier

### Description

This is an implementation of the integrating information gain algorithm(ig) into tree based boosting for two-class classification problem. 
The ig is used to select sub set features of data. Due to the imbalanced property of our data sets, we add a bagging structure to solve the provlem.
In boosting process, the algorithm sequentially applies a weak classification result to modified versions of the data. By increasing the weights of 
the mis-classified observations, each weak learner focuses on the error of the previous one. We assign the weights to all the data (by multiply), 
and in those "modified data" we run a feature selection method(using information gain) to select top-ranked features. The predictions are aggregated 
through a weighted majority vote. In bagging process, the input negative data (majority data) sets are under-sampled to balance data distribution.
The final predictions  of bagging are also aggregated through majority vote.

### Method 
Information Gain based Bagging Boosting algorithm:<br />
<img src="https://github.com/designer357/IGBB/blob/master/images/igbb.png"> <br />
### Data 
The original BGP data set is from RIPE Network Coordination Center: [RIPE RIS raw data] (https://www.ripe.net/analyse/internet-measurements/routing-information-service-ris/ris-raw-data)  
### Results
Using the Hastie (10.2) dataset, we can appreciate a significant reduction in the error rate as we increase the number of iterations. <br />
<img src="https://github.com/jaimeps/adaboost-implementation/blob/master/images/error_rate.png" width="500"> <br />


### References
- Trevor Hastie, Robert Tibshirani, Jerome Friedman - *The Elements of Statistical Learning*


If had any problem, please send me email: mc.cheng@my.cityu.edu.hk
### License
This project is licensed under the MIT License.