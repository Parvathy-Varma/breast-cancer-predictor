'''Types of tumor
Bengin tumor : 
  Non-cancerous
  Capsulated - it's enclosed in a fibrous sheath of connective tissue, forming a distinct boundary that keeps it separate from surrounding normal tissue, making it slow-growing, localized, and generally easier to remove surgically without invasion, unlike cancerous tumors which spread
  Non-invasive - doesn't invade or destroy surrounding healthy tissues
  Slow growing
  Do not metastasize (spread to other parts of the body)
  Cells are normal

Malignant tumor:
  Cancerous
  Non-capsulated
  Fast growing
  Metastasize (spread) to other parts of the body
  Cells have large, dark nucle may have abnormal shape

'''
'''
The dataset we are going to use is from a fine needle aspiration
Fine needle aspiration is a type of biopsy procedure. 
In fine needle aspiration, a thin needle is inserted into an area of abnormal-appearing tissue or body fluid.
As with other types of biopsies, the sample collected during fine needle aspiration can help make a diagnosis 
or rule out conditions such as cancer.

It is a labelled dataset 0 represents bengin and 1 represents maligant
we need to classify the data
we use logistic regression model for binary classification 0 or 1
'''

'''Import the dependencies'''
import numpy as np
import pandas as pd
import sklearn.datasets #we can get the datset from here or even from kaggle data.csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

'''Data collection and processing'''
#loading data from sklearn library to the variable breast_cancer_dataset
breast_cancer_dataset=sklearn.datasets.load_breast_cancer()
#this will be in the form of numpy n dimensional array when we try to print it
#loading the data to a pandas dataframe
df=pd.DataFrame(breast_cancer_dataset.data,columns=breast_cancer_dataset.feature_names)
#the data consists of "data" key with its values ,"target" key with its target values and "feature names" ky with its column names values
#we need to get all those values
#adding the "target" columns to the df
df['label']=breast_cancer_dataset.target #we are adding a new column to the df coz the df does not contain the target values (0 and 1)
#we created a new column label and loaded the values from the array target of the dataset
#check whether the df is fine by using haed() or tail()
#df.shape will give us the number of columns and rows 
#df.shape returns a tuple so we should not write df.shape()
#there are 569 rows and 31 columns, among this one column is the output
#getting some information about the data: df.info() we can know the data type, missing values etc
#if there are any missing values we should use imputation tachniques and solve that like finding mean of all other values etc
#we can also check missing values using: df.isnull().sum() all the columns will be mentioned with its number of missing values
#to get statistical measures: df.describe() we will have certain percentile values there not percentage
#25% percentile means 25 percentage of data are less than tha particular value mentioned in the row
#checking the distribution of target variabl how many 0s and 1s : df['label'].value_counts()
#find mean of all columns of '0' values and of all columns of '1' values to compare the values of 0 and 1
# df.groupby('label').mean(): from this we understood that the mean value of each column of 0 is greater than that of 1
# separating features and labels
x=df.drop(columns='label',axis=1)
y=df['label']


'''Splitting data into training data and test data'''
#we will have 4 numpy arrays
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=2)
#check the number of columns and rows using shape


'''Model training'''
#we are using logistic regression which is very useful for binary classification
model=LogisticRegression() #loading an instance of the logistic regression model into the variable model
#fit the dataset to the logistic regression model
#training the logistic regression
model.fit(X_train,Y_train) #it will try to figure out the relationship between the x and y values
#how data looks like if label 0 or label 1
#goes through the data repeatedly and tries to find the best fit for the data


'''Model evaluation'''
#accuracy on training data
#we will predict the label values for x_train data and compare it with y_train and c the accuracy
X_train_prediction=model.predict(X_train) #all predicted labels of x_train will be in x_train_prediction variable
#we will compare x_train_prediction values with the true values y_train
training_data_accuracy=accuracy_score(Y_train,X_train_prediction)
#print training_data_accuracy to see the value of accuracy of training data
print("Accuracy on training data=",training_data_accuracy)
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print("Accuracy of test data=",test_data_accuracy)
#sometimes the model gets overfit
#this means that it learns the training data more and thus the accuracy score from training data will be more
#the accuracy score of test data will be less in case of overfitting so it is important to test the accuracy of both training data and test data


'''Building a predictive system'''
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
input_data=(17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)
#convert to array so we can fit into our model
input_data_as_np_array=np.asarray(input_data)
#reshape the numpy array as we are predicting for one data point
input_data_reshape=input_data_as_np_array.reshape(1,-1)
prediction=model.predict(input_data_reshape)
if prediction[0]==0:
    print("The cancer is Malignant")
else:
    print("The cancer is Bengin")