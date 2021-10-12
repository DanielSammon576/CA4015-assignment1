#!/usr/bin/env python
# coding: utf-8

# # **Whole dataset clustering**

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from numpy import arange

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


clustering = pd.read_csv('Data\whole_clustering.csv')
cluster = KMeans(n_clusters = 4)
cols = clustering.columns[:]
clustering.drop(clustering.columns[[0]], axis = 1, inplace = True)
clustering.head()


# So for my clustering analysis of the whole study I would like to see if the studies as a whole have interesting cluster patterns. To do this step I will have to map an integer value to the study so that clustering can take place.

# In[4]:


y_predicted = cluster.fit_predict(clustering[["Margin","count_zeros"]])
clustering["cluster"] = y_predicted
clustering.head()


# In[5]:


df1 = clustering[clustering.cluster==0]
df2 = clustering[clustering.cluster==1]
df3 = clustering[clustering.cluster==2]
df4 = clustering[clustering.cluster==3]

plt.scatter(df1.Margin, df1.count_zeros, color='green')
plt.scatter(df2.Margin, df2.count_zeros, color='red')
plt.scatter(df3.Margin, df3.count_zeros, color='black')
plt.scatter(df4.Margin, df4.count_zeros, color='blue')

plt.xlabel("Margin")
plt.ylabel("count_zeros")
plt.legend()


# In[7]:


clustering[['Margin','count_zeros']] = minmax_scale(clustering[['Margin','count_zeros']])
clustering.head()


# In[8]:


km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(clustering[["Margin", "count_zeros"]])


# In[9]:


clustering["cluster"] = y_predicted
clustering.head()


# In[10]:


km.cluster_centers_


# In[11]:


df1 = clustering[clustering.cluster==0]
df2 = clustering[clustering.cluster==1]
df3 = clustering[clustering.cluster==2]
df4 = clustering[clustering.cluster==3]

plt.scatter(df1.Margin, df1.count_zeros, color='green')
plt.scatter(df2.Margin, df2.count_zeros, color='red')
plt.scatter(df3.Margin, df3.count_zeros, color='black')
plt.scatter(df4.Margin, df4.count_zeros, color='blue')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color="orange", marker="*", label="centroid")

plt.xlabel("Margin")
plt.ylabel("count_zeros")
plt.legend()


# In[12]:


k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(clustering[["Margin", "count_zeros"]])
    sse.append(km.inertia_)


# In[13]:


sse


# In[14]:


plt.xlabel("K")
plt.ylabel("Sum of squared error")
plt.plot(k_rng, sse)


# In[18]:


sns.scatterplot(data=clustering, x="Margin", y="count_zeros", hue="Study")
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

