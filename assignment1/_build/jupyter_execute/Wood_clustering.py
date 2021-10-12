#!/usr/bin/env python
# coding: utf-8

# # **Wood study clustering**

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from numpy import arange

import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

import sklearn.metrics as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


clustering = pd.read_csv('Data\clustering.csv')
cluster = KMeans(n_clusters = 4)
cols = clustering.columns[:]
clustering.drop(clustering.columns[[0]], axis = 1, inplace = True)
clustering.head()


# In[3]:


y_predicted = cluster.fit_predict(clustering[["Difference","Total-B/D"]])
clustering["cluster"] = y_predicted
clustering.head()


# In[4]:


df1 = clustering[clustering.cluster==0]
df2 = clustering[clustering.cluster==1]
df3 = clustering[clustering.cluster==2]
df4 = clustering[clustering.cluster==3]

plt.scatter(df1.Difference, df1["Total-B/D"], color='green')
plt.scatter(df2.Difference, df2["Total-B/D"], color='red')
plt.scatter(df3.Difference, df3["Total-B/D"], color='black')
plt.scatter(df4.Difference, df4["Total-B/D"], color='blue')

plt.xlabel("Difference")
plt.ylabel("Total-B/D")
plt.legend()


# In[5]:


clustering[['Difference','Total-B/D']] = minmax_scale(clustering[['Difference','Total-B/D']])
clustering.head()


# In[6]:


km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(clustering[["Difference", "Total-B/D"]])


# In[7]:


clustering["cluster"] = y_predicted
clustering.head()


# In[8]:


km.cluster_centers_


# In[9]:


df1 = clustering[clustering.cluster==0]
df2 = clustering[clustering.cluster==1]
df3 = clustering[clustering.cluster==2]
df4 = clustering[clustering.cluster==3]

plt.scatter(df1.Difference, df1["Total-B/D"], color='green')
plt.scatter(df2.Difference, df2["Total-B/D"], color='red')
plt.scatter(df3.Difference, df3["Total-B/D"], color='black')
plt.scatter(df4.Difference, df4["Total-B/D"], color='blue')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color="orange", marker="*", label="centroid")

plt.xlabel("Difference")
plt.ylabel("Total-B/D")
plt.legend()


# In[10]:


k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(clustering[["Difference", "Total-B/D"]])
    sse.append(km.inertia_)


# In[11]:


sse


# In[12]:


plt.xlabel("K")
plt.ylabel("Sum of squared error")
plt.plot(k_rng, sse)
#This is indictaing that the optimum number of clusters is 4 but 3 is also a reasonable choice


# In[14]:


sns.scatterplot(data=clustering, x="Difference", y="Total-B/D", hue="AgeProfile")
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


# In[ ]:




