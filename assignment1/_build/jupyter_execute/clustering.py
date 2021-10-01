#!/usr/bin/env python
# coding: utf-8

# # **Clustering**

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from numpy import arange

import sklearn
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

import sklearn.metrics as sm
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


#Focusing on 100 choices from 200 subjects as a first step
choice = pd.read_csv('IGTdataSteingroever2014\choice_150.csv')


# In[3]:


#All data types are the same and the original matrix is clean
choice.dtypes


# In[4]:


cluster = KMeans(n_clusters = 5)
cols = choice.columns[:]
cols


# In[5]:


choice["cluster"] = cluster.fit_predict(choice[choice.columns[1:]])
choice.head()
choice.tail()


# In[6]:


pca = PCA(n_components = 2)
choice["x"] = pca.fit_transform(choice[cols])[:,0]
choice["y"] = pca.fit_transform(choice[cols])[:,1]
choice = choice.reset_index()
choice.tail()


# In[7]:


df = choice[["index", "cluster", "x", "y"]]
df.columns = ["Subjects", "Cluster", "X", "Y"]
df.head()


# In[8]:


import seaborn as sns
sns.scatterplot(data=df,x="X", y="Y", hue="Cluster")


# In[9]:


#Now that we have looked at the different choices that have been made we can take a look at the results
win = pd.read_csv('IGTdataSteingroever2014\wi_150.csv')
loss = pd.read_csv("IGTdataSteingroever2014\lo_150.csv")
index = pd.read_csv("IGTdataSteingroever2014\index_150.csv")


# In[10]:


win.tail(5)


# In[11]:


loss.tail(5)


# In[12]:


choice.tail(5)


# In[13]:


choice_new = choice.apply(pd.Series.value_counts, axis=1)
choice_new["SubjectId"] = choice["index"]
cols = list(choice_new.columns)
cols = [cols[-1]] + cols[:-1]
choice_new = choice_new[cols]
choice_new.drop(choice_new.iloc[:, 5:], inplace = True, axis = 1)
choice_new.columns = ["Subject_id", "A", "B", "C", "D"]
choice_new.fillna(value = 0, inplace = True)
choice_new.A = choice_new.A.astype(int)
choice_new.B = choice_new.B.astype(int)
choice_new.C = choice_new.C.astype(int)
choice_new.D = choice_new.D.astype(int)
choice_new.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




