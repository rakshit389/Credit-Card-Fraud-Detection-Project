import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
cred=pd.read_csv('creditcard.csv')
cred
cred.isnull().sum()
cred.columns
cred.info()
cred.describe().transpose()
plt.figure(figsize=(30,20))
sns.heatmap(cred.corr(),annot=True,cmap='YlOrRd')
plt.show()
cred.hist(figsize=(20,20))
plt.show()
cred_=cred.iloc[:,1:29]
cred_hd=cred_.head(20)
cred_hd.plot(kind='hist',figsize=(16,8),grid=True)
plt.title('Histogram of (V1-V28) Top 20 Data')
plt.xlabel('Values of (V1-V28)')
plt.ylabel('Number of (V1-V28) Data')
sns.set(rc={'figure.figsize':(20,12)})
sns.distplot(cred_hd)
plt.title('Distplot of (V1-V28) Top 20 Data')
plt.xlabel('Values of (V1-V28)')
plt.ylabel('Number of (V1-V28) Data')
cred_=cred.iloc[:,1:29]
cred_hd=cred_.head(15)
cred_hd.plot(kind='line',figsize=(20,10),grid=True)
plt.title('Line Plot of (V1-V28) Top 15 Data')
plt.xlabel('Values of (V1-V28)')
cred_=cred.iloc[:,1:29]
cred_hd=cred_.head(20)
cred_hd.plot(kind='box',figsize=(20,10),grid=True)
plt.title('Box Plot of (V1-V28) Top 20 Data')
plt.ylabel('Values of (V1 - V28)')
cred_=cred.iloc[:,1:11]
cred_hd=cred_.head(15)
cred_hd.plot(kind='bar',figsize=(17,7),grid=True)
plt.title('Bar Plot of (V1-V10) Top 15 Data')
plt.figure(figsize=(10,6))
sns.regplot(x='Class',y='Amount',data=cred,color='black')
plt.title('Regression Plot of Class Vs Amount')
plt.show()
plt.figure(figsize=(10,6))
sns.regplot(x='Amount',y='Time',data=cred)
plt.title('Regression Plot of Amount Vs Time')
plt.show()
plt.figure(figsize=(10,6))
sns.regplot(x='Amount',y='V1',data=cred,color='red')
plt.title('Regression Plot of Amount Vs V1')
plt.show()
plt.figure(figsize=(10,6))
sns.regplot(x='Amount',y='V2',data=cred,color='green')
plt.title('Regression Plot of Amount Vs V2')
plt.show()
sns.countplot(cred['Class'],palette='magma')
plt.title('Valid Data Vs Fraud Data')
plt.show()
fraud_dt=cred[cred['Class']==1]
valid_dt=cred[cred['Class']==0]

print('Fraud Cases:',len(fraud_dt))
print('Valid Cases:',len(valid_dt))

outlier_fraction=len(fraud_dt)/len(valid_dt)
outlier_fraction
plt.ylabel('Number of (V1-V28) Data')
x=cred.iloc[:,1:29]
x
y=cred.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.25)
print('Train Size = {} \nTest Size = {} \nTotal Size = {}'.format(x_train.shape[0],x_test.shape[0],x.shape[0]))
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100)
rfc
RandomForestClassifier()
rfc.fit(x_train,y_train)
RandomForestClassifier()
y_pred=rfc.predict(x_test)
y_pred
array([1, 0, 0, ..., 0, 0, 0], dtype=int64)
rfc_score=rfc.score(x_test,y_test)*100
print("Accuracy of the Credit Card data :",rfc_score)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
sns.set(rc={'figure.figsize':(8,6)})
sns.heatmap(cm,annot=True,cmap='YlOrRd')

Accuracy = (71082+87)/(71082+26+7+87)
print('Accuracy Score by formula is:',Accuracy)
Accuracy Score by formula is: 0.9995365298727564
from sklearn.metrics import accuracy_score
acc_score=accuracy_score(y_test,y_pred)
print(f'Accuracy of the Model: {(acc_score*100)//1} %')

#1. Recall Score by formula for 0 is : (TP/(TP+FN)): 
print('Recall Score by formula for 0 is:',(71082/(71086+26)))
#2. Recall Score by formula for 1 is : (TP/(TP+FN)): 
print('Recall Score by formula for 1 is:',(87/(87+7)))

#3. Precision Score by formula for 0 is : (TP/(TP+FP)): 
print('Precision Score by formula for 0 is:',(71082/(71086+87)))
#4. Precision Score by formula for 1 is : (TP/(TP+FP)): 
print('Precision Score by formula for 1 is:',(87/(87+26)))

from sklearn.metrics import recall_score,precision_score
prec_score=precision_score(y_test,y_pred)
print(f'Precision Score of the Model: {(prec_score*100)//1} %')
rec_score=recall_score(y_test,y_pred)
print(f'Recall Score of the Model: {(rec_score*100)//1} %')
