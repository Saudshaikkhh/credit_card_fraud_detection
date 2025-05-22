import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
import joblib

creditdata = pd.read_csv('creditcard.csv')
# print(creditdata)
# print(creditdata.head(5)) #checking the values of first five 


#dataset information 
# print(creditdata.info()) checkiing their datatypes


#checking the values of missing numbers in each column
# print(creditdata.isnull().sum()) #no missing value


#distrubution of legit transaction and fraudulent transaction
# print(creditdata['Class'].value_counts()) # {0: 284315; 1: 492}
#this dataset is highly unbalanced 

#seperating the data for the analysis
legit = creditdata[creditdata.Class ==0]
fraud = creditdata[creditdata.Class ==1]
# print(legit.shape)
# print(fraud.shape)

#now we will go through the process name undersampling
#as the was unbalanced so now we gonna build a sample dataset containing similar distribution of normal transaction & fraud trans
# no of f trans = 492 so what we gonna do is we gonna take random values form legit anf it sixe should be as same as fraud as the 
# fraud as a small size as compared to legit so it will be a challenge if the one differenciation consist more a lot data than one 
# so we gonna take random values from the legit and gonna create a sample dataset

legit_sample = legit.sample(n=492)
#concatenating two  dataframe and creating a new dataframe
ndataset = pd.concat([legit_sample, fraud], axis=0)
# print(ndataset.head(5))
# print(ndataset.shape) #984

#splitting the data into features and targets
x = ndataset.drop(columns='Class',axis=1)
y = ndataset['Class']

#splitting features and targets into training an test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

#model training with training data
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
joblib.dump(model, 'creditcard_fraud_model.pkl')

#model evaluation

#accuracy score
#accuracy on training data
x_train_prediction = model.predict(x_train)
tdaccuracy = accuracy_score(x_train_prediction,y_train)
# print(tdaccuracy) #printed the accuracy is 0.9326556543837357

#accuracy on test data
x_test_prediction = model.predict(x_test)
ttdaccuracy = accuracy_score(x_test_prediction,y_test)
print(ttdaccuracy)