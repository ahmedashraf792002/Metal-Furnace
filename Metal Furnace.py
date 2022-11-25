#Import Libraries
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
#read data
data=pd.read_csv('Train2.csv')
#print 5 frist data
print(data.head())
#print data columns
x=1
for col in data.columns:
    print(f'column {x} is : ',col)
    x+=1
#data corr
print(data.corr()) 
sns.heatmap(data.corr())
#data info
print(data.info())   
#show data contain null data
print(data.isnull().sum())
#show data contain dublicate data
print(data.duplicated().sum())
#data describe
print(data.describe())
#show input X and output y
col=data.shape[1]
X=data.iloc[:,0:col-1]
y=data.iloc[:,col-1:col]
print(col)
print(X.shape)
print(y.shape)
#data spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)
#Applying LogisticRegression Model 
LogisticRegressionModel = LogisticRegression(penalty='l2',solver='sag',C=1,random_state=33)
LogisticRegressionModel.fit(X_train, y_train)
#Calculating Details
print('LogisticRegressionModel Train Score is : ' , LogisticRegressionModel.score(X_train, y_train))
print('LogisticRegressionModel Test Score is : ' , LogisticRegressionModel.score(X_test, y_test))
print('LogisticRegressionModel Classes are : ' , LogisticRegressionModel.classes_)
print('LogisticRegressionModel No. of iteratios is : ' , LogisticRegressionModel.n_iter_)
#Calculating Prediction
y_pred = LogisticRegressionModel.predict(X_test)
y_pred_prob = LogisticRegressionModel.predict_proba(X_test)
print('Predicted Value for LogisticRegressionModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for LogisticRegressionModel is : ' , y_pred_prob[:10])
#Applying RandomForestClassifier Model 
RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini',n_estimators=100,max_depth=20,random_state=33)
RandomForestClassifierModel.fit(X_train, y_train)
#Calculating Details
print('RandomForestClassifierModel Train Score is : ' , RandomForestClassifierModel.score(X_train, y_train))
print('RandomForestClassifierModel Test Score is : ' , RandomForestClassifierModel.score(X_test, y_test))
print('RandomForestClassifierModel features importances are : ' , RandomForestClassifierModel.feature_importances_)
print('RandomForestClassifierModel Classes are : ' , RandomForestClassifierModel.classes_)
#display RandomForestClassifierModel.feature_importances_
plt.figure()
X_bar=range(0,28)
x_bar=list(X_bar)
plt.bar(X_bar,RandomForestClassifierModel.feature_importances_)
#Calculating Prediction
y_pred = RandomForestClassifierModel.predict(X_test)
y_pred_prob = RandomForestClassifierModel.predict_proba(X_test)
print('Predicted Value for RandomForestClassifierModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for RandomForestClassifierModel is : ' , y_pred_prob[:10])
#Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)
# drawing confusion matrix
plt.figure()
sns.heatmap(CM, center = True)
#Calculating Confusion Matrix
CM = confusion_matrix(y_train,RandomForestClassifierModel.predict(X_train))
print('Confusion Matrix is : \n', CM)
# drawing confusion matrix
plt.figure()
sns.heatmap(CM, center = True)
#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))
AccScore = accuracy_score(y_test, y_pred, normalize=True)
print('Accuracy Score is : ', AccScore)
#Calculating F1 Score  : 2 * (precision * recall) / (precision + recall)
F1Score = f1_score(y_test, y_pred, average='micro') 
print('F1 Score is : ', F1Score)
#Calculating Precision Score : (Specificity) #(TP / float(TP + FP))  
PrecisionScore = precision_score(y_test, y_pred, average='micro')
print('Precision Score is : ', PrecisionScore)
#read test data
test_data=pd.read_csv('Test1.csv')
#print 5 frist test data
print(test_data.head())
#print test data columns
x=1
for col in test_data.columns:
    print(f'column {x} is : ',col)
    x+=1
#predict test_data    
submision=RandomForestClassifierModel.predict(test_data)  
#predict test_data is csv
submision=pd.DataFrame({'output':submision})
submision.to_csv('submision.csv')
  