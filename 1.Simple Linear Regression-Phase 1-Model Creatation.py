#!/usr/bin/env python
# coding: utf-8

# In[3]:


#pandas is a library and it contains the functions related to the database
import pandas as pd


# In[4]:


#This line of code reads data from a CSV file named "Salary_Data.csv" into a Pandas variable called dataset.
dataset=pd.read_csv("Salary_Data.csv")


# In[5]:


#This command gives the data that is stored under the variable dataset
dataset


# In[6]:


#This line creates a new variable named "independent"containing only the "YearsExperience" column from the original Data "dataset".
independent=dataset[["YearsExperience"]]
independent


# In[9]:


#This line creates a new variable named "independent"containing only the "Salary" column from the original Data "dataset".
dependent=dataset[["Salary"]]
dependent


# In[11]:


#This line splits the data into train and test sets for independent and dependent variables with a test size of 30% and a specified random state from sklearn libraury
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(independent, dependent, test_size=0.30,random_state=0)


# In[12]:


#This line creates a Linear Regression model object named "regressor" and fits it to the training data.
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[14]:


#this code line retries the weight calculated from the excecuted trained model in linear regression
weight=regressor.coef_
weight


# In[15]:


##this code line retries the bias calculated from the excecuted trained model in linear regression
bais=regressor.intercept_
bais


# In[16]:


#This line generates predictions for the target variable using the trained linear regression model and the testing data
y_pred=regressor.predict(X_test)


# In[17]:


#this line gives the r2 value from y_train and y_pred to find whether the model is good or poor model
from sklearn.metrics import r2_score
r_score=r2_score(y_test,y_pred)


# In[18]:


r_score


# In[19]:


#picle is a libraury to save the model with the file name "finalized_model_linear.sav" and .Sav is a extention
import pickle
filename="finalized_model_linear.sav"


# In[20]:


#this line of code saves the trained model in a serialised format
pickle.dump(regressor,open(filename,'wb'))


# In[23]:


#This line loads a trained linear regression model from a saved file and uses it to make predictions on new data, in this case, predicting the target variable value for a new input of 13
loaded_model=pickle.load(open("finalized_model_linear.sav",'rb'))
result=loaded_model.predict([[13]])


# In[24]:


result


# In[ ]:




