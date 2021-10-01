#!/usr/bin/env python
# coding: utf-8

# # **Data Analysis**

# 

# ### **Introduction to Data**

# The data set at hand is divided into three different trials 95-trial, 100-trial and a 150-trial. There is three seperate csv files per trial. Let's take the 95-trial we have a csv file that records the participants choices, a csv file that records the participants losses and a csv file that records the participants winnings.

# As all of the data isn't gathered from one study it is in fact gathered from 10 seperate studies we are also given a fourth csv file which maps what study each participant took part in.

# The studies differ in many ways from the size of the actual trials to the age demographics of the studies.

# 

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
choice = pd.read_csv('IGTdataSteingroever2014\choice_100.csv')
win = pd.read_csv('IGTdataSteingroever2014\wi_100.csv')
loss = pd.read_csv("IGTdataSteingroever2014\lo_100.csv")
index = pd.read_csv("IGTdataSteingroever2014\index_100.csv")


# Below shows us the total times a participant choose a certain card. For example Subj_3 picked "D" 88 times but only choose "C"

# In[3]:


choice_new = choice.apply(pd.Series.value_counts, axis=1)
choice_new.columns = ["A", "B", "C", "D"]
choice_new.fillna(value = 0, inplace = True)
choice_new.A = choice_new.A.astype(int)
choice_new.B = choice_new.B.astype(int)
choice_new.C = choice_new.C.astype(int)
choice_new.D = choice_new.D.astype(int)
choice_new["Study"] = index["Study"].values


# The Wood study was divided up in age. The first 90 participants were between the ages of 18-40 with the remaining 62 participants between the ages of 61-88. My proposal is to look at the difference between the two age groups and see whether the younger participants were quicker to identify the beneficial cards.

# In[4]:


choice_new["Total-B/D"] = choice_new["B"] + choice_new["D"]


# In[5]:


young = choice_new[0:91]
a = young["A"].sum()
b = young["B"].sum()
c = young["C"].sum()
d = young["D"].sum()


data = [{'A': a, 'B': b, 'C':c, "D":d}]
young_stats = pd.DataFrame(data)
young_stats = young_stats.rename(index={0: 'Counts'})


# In[6]:


old = choice_new[91:]
a = old["A"].sum()
b = old["B"].sum()
c = old["C"].sum()
d = old["D"].sum()

data = [{'A': a, 'B': b, 'C':c, "D":d}]
old_stats = pd.DataFrame(data)
old_stats = old_stats.rename(index={0: 'Counts'})


# In[7]:


win['Total'] = win.sum(axis=1)
loss['Total'] = loss.sum(axis=1)


# In[8]:


subject = pd.DataFrame(columns=["Subjects"])
subject["Subjects"] = win.index
subject["Difference"] = win["Total"].values + loss["Total"].values
subject["Total-B/D"] = choice_new["Total-B/D"].values/100 * 100
subject["Study"] = index["Study"].values
subject = subject[subject.Study == "Wood"]
subject["AgeProfile"] = ""
subject.AgeProfile.values[:91] = "Young"
subject.AgeProfile.values[91:] = "Old"


# In[9]:


sns.scatterplot(data=subject, x="Difference", y="Total-B/D", hue="AgeProfile")


# The dataset had a larger representation of younger people, from the scatter plot above you can clearly see that there are more older nodes making extreme loses and percentage wise more older people losing money. From my initial data analysis it seems as though in general the younger people were quicker to figure out that the B card and the D card were

# In[ ]:




