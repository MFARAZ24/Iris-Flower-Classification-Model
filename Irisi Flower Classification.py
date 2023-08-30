#!/usr/bin/env python
# coding: utf-8

# # Data Science Internship
# 
# # Task 01: Iris Flower Classification Model
# 
# # M.Faraz Shoaib
# 
# ## Importing Libraries and Data
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
fl_data = pd.read_csv ("IRIS.csv")


# In[2]:


print(fl_data.head())


# In[3]:


print(fl_data.describe())


# In[4]:


print("Target labels", fl_data['species'].unique())


# ## Plotting Data to Check Relation
# 

# In[5]:


import plotly.express as px
fig = px.scatter(fl_data, x = "sepal_width", y = "sepal_length", color = "species")
fig.show()


# ## Spliting Data for Train and Test

# In[6]:


x = fl_data.drop("species", axis=1)
print(x)
y = fl_data["species"]
print(y)


# ## Training Model 

# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train,y_train)


# ## Evaluation of Model

# In[8]:


new_val = np.array([[6,3,4,1]])
predct = model.predict(new_val)
print("predct :{}".format(predct))

