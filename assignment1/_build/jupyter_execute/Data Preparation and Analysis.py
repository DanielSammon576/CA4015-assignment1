#!/usr/bin/env python
# coding: utf-8

# # Data Preparation and Analysis

# 

# ### Introduction to Data

# The data set at hand is divided into three different trials 95-trial, 100-trial and a 150-trial. There is three seperate csv files per trial. Let's take the 95-trial - we have a csv file that records the participants choices, a csv file that records the participants losses and a csv file that records the participants winnings.

# As all of the data is not gathered from one study but is in fact gathered from 10 seperate studies, to handle this we are given a fourth csv file which maps what study each participant took part in.

# The studies differ in many ways from the size of the actual trials to the age demographics of the studies.

# **Libraries used**

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns

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
choice95 = pd.read_csv('Data\choice_95.csv')
win95 = pd.read_csv('Data\wi_95.csv')
loss95 = pd.read_csv("Data\lo_95.csv")
index95 = pd.read_csv("Data\index_95.csv")
choice100 = pd.read_csv('Data\choice_100.csv')
win100 = pd.read_csv('Data\wi_100.csv')
loss100 = pd.read_csv("Data\lo_100.csv")
index100 = pd.read_csv("Data\index_100.csv")
choice150 = pd.read_csv('Data\choice_150.csv')
win150 = pd.read_csv('Data\wi_150.csv')
loss150 = pd.read_csv("Data\lo_150.csv")
index150 = pd.read_csv("Data\index_150.csv")


# In[3]:


zeros95 = loss95.isin([0]).sum(axis=1)
zeros95 = pd.DataFrame(zeros95)
zeros95.columns = ["zeros"]

zeros100 = loss100.isin([0]).sum(axis=1)
zeros100 = pd.DataFrame(zeros100)
zeros100.columns = ["zeros"]

zeros150 = loss150.isin([0]).sum(axis=1)
zeros150 = pd.DataFrame(zeros150)
zeros150.columns = ["zeros"]


# In[4]:


df95 = pd.DataFrame()
df100 = pd.DataFrame()
df150 = pd.DataFrame()

df95["Total W"] = win95.sum(axis=1)
df95["Total L"] = loss95.sum(axis=1)

df100["Total W"] = win100.sum(axis=1)
df100["Total L"] = loss100.sum(axis=1)

df150["Total W"] = win150.sum(axis=1)
df150["Total L"] = loss150.sum(axis=1)

df95.reset_index(inplace=True)
df100.reset_index(inplace=True)
df150.reset_index(inplace=True)

df95["Study"] = index95["Study"].values
df100["Study"] = index100["Study"].values
df150["Study"] = index150["Study"].values

df95["Margin"] = df95["Total W"] + df95["Total L"]
df100["Margin"] = df100["Total W"] + df100["Total L"]
df150["Margin"] = df150["Total W"] + df150["Total L"]

df95["count_zeros"] = zeros95["zeros"].values
df100["count_zeros"] = zeros100["zeros"].values
df150["count_zeros"] = zeros150["zeros"].values

df95.size + df100.size + df150.size #2468

final = pd.DataFrame()
alternative = pd.DataFrame()
alternative = df95.append(df100)
final = alternative.append(df150)
final.size #2468
final.head()


# ### Data visualisation

# In[5]:


sns.scatterplot(data=final, x="Total W", y="Total L", hue="Study")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# In[6]:


sns.barplot(x="Study", y="Margin", data=final)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.xticks(rotation=45)


# With the scatter matrix plot below I hope the relationships between the different variables will provide me with some interesting insights.

# In[7]:


pd.plotting.scatter_matrix(final[["Total W", "Total L", "Margin", "count_zeros"]], figsize=(12.5,12.5), hist_kwds=dict(bins=35))
plt.show()


# ### Observations and wood study visualisations

# From my above data analysis I can see one study in particular whose margins were surprising. The Wood study in both graphs shows that participants were making considerable losses. Upon inspection this study was ran on two different groups of people. The first 90 participants were between the ages of 18-40 with the remaining 62 participants between the ages of 61-88. My proposal is to look at the difference between the two age groups and see whether the younger participants were quicker to identify the beneficial cards.

# In[8]:


choice_new = choice100.apply(pd.Series.value_counts, axis=1)
choice_new.columns = ["A", "B", "C", "D"]
choice_new.fillna(value = 0, inplace = True)
choice_new.A = choice_new.A.astype(int)
choice_new.B = choice_new.B.astype(int)
choice_new.C = choice_new.C.astype(int)
choice_new.D = choice_new.D.astype(int)
choice_new["Study"] = index100["Study"].values


# In[9]:


choice_new["Total-B/D"] = choice_new["B"] + choice_new["D"]


# In[10]:


young = choice_new[0:91]
a = young["A"].sum()
b = young["B"].sum()
c = young["C"].sum()
d = young["D"].sum()


data = [{'A': a, 'B': b, 'C':c, "D":d}]
young_stats = pd.DataFrame(data)
young_stats = young_stats.rename(index={0: 'Counts'})


# In[11]:


old = choice_new[91:]
a = old["A"].sum()
b = old["B"].sum()
c = old["C"].sum()
d = old["D"].sum()

data = [{'A': a, 'B': b, 'C':c, "D":d}]
old_stats = pd.DataFrame(data)
old_stats = old_stats.rename(index={0: 'Counts'})


# In[12]:


win100['Total'] = win100.sum(axis=1)
loss100['Total'] = loss100.sum(axis=1)


# 

# The subject dataframe I will use to cluster only the Wood study. This study was ran on two seperate groups with different ages so will hopefully provide interesting results.

# In[13]:


subject = pd.DataFrame(columns=["Subjects"])
subject["Subjects"] = win100.index
subject["Difference"] = win100["Total"].values + loss100["Total"].values
subject["Total-B/D"] = choice_new["Total-B/D"].values/100 * 100
subject["Study"] = index100["Study"].values
subject = subject[subject.Study == "Wood"]
subject["AgeProfile"] = ""
subject.AgeProfile.values[:91] = "Young"
subject.AgeProfile.values[91:] = "Old"
print("Subject dataframe")
subject.head(10)


# In[14]:


sns.barplot(x="Subjects", y="Difference", data=subject, hue="AgeProfile")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax1 = plt.axes()
x_axis = ax1.axes.get_xaxis()
x_axis.set_visible(False)
plt.show()
plt.close()


# In[15]:


pd.plotting.scatter_matrix(subject[["Difference", "Total-B/D"]], figsize=(12.5,12.5), hist_kwds=dict(bins=35))
plt.show()


# The dataset had a larger representation of younger people, using the dataframe above I will inspect the difference between younger and older both in profit margins and how quick the two age groups were to realise that some cards are more benficial then others. I use different analysis techniques including scatter graphs and k-means clustering to evaluate this hypothesis.

# In[16]:


#This is the dataset that we will be using for our clustering of the wood study
subject.to_csv("Data/clustering.csv")

#This is the dataset we will be using for the whole study clustering
final.to_csv("Data/whole_clustering.csv")

