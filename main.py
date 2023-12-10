#Import dependencies 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

######### Data collection #########
pkdata = pd.read_csv(r"C:\.Projects\Parkinson\parkinsons.csv")
pkdata.head()
pkdata.shape
pkdata.info()
pkdata.isnull().sum()
pkdata.describe()
pkdata["status"].value_counts()

#separate feats and target
x=pkdata.drop(columns=["name","status"], axis=1)
y=pkdata["status"]
#print(y)

numeric_columns = pkdata.select_dtypes(include=np.number).columns
pknumeric = pkdata[numeric_columns]
pkdata_grouped = pknumeric.groupby("status").mean()
#print(pkdata_grouped)


#split into 2 groups 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
#print(x.shape,x_train.shape, x_test.shape)

#data standarization 
scaler=StandardScaler()
scaler.fit(x_train)

x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
#print(x_train)

######### Model training #########

#Support vector machine model
model=svm.SVC(kernel="linear")
# Training SVMmodel w training data
model.fit(x_train, y_train)


######### Evaluating the model #########
## Check accuracy score of training data
x_train_prediction=model.predict(x_train)
trdataacc=accuracy_score(y_train, x_train_prediction)
#print("accuracy_score of training data is ", trdataacc)

##Check accuracy score of test data
x_test_prediction=model.predict(x_test)
testacc=accuracy_score(y_test, x_test_prediction)
#print("accuracy_score of test data is ", testacc )

########## Building a predictive system #########
# We ask the user to input data --> you can try with any subject from parkinson.csv
input_data_str = input("Please enter the person's measurements separated by commas: ")

# Convert input data to a numpy array
input_data_array = np.array([float(value) for value in input_data_str.split(',')])

# Reshape and standardize the input data
input_data_reshape = input_data_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshape)

# Make a prediction
pred = model.predict(std_data)
if pred[0] == 0:
    print("The person doesn't have Parkinson Disease")
else:
    print("The person has Parkinson Disease")
