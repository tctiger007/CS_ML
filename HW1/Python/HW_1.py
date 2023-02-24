#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as mp
from pylab import show


# In[2]:


data = np.loadtxt("data.csv")
np.random.seed(100)
np.random.shuffle(data)
features = []
digits = []
for row in data:
    if(row[0]==1 or row[0]==5):
        features.append(row[1:])
        digits.append(str(row[0]))
numTrain = int(len(features)*0.2)
trainFeatures = features[:numTrain]
testFeatures = features[numTrain:]
trainDigits = digits[:numTrain]
testDigits = digits[numTrain:]


# In[6]:


X = []   ##mean
Y = []   ##std
colors = []
for index in range(len(trainFeatures)):
    X.append(np.mean(trainFeatures[index]))
    Y.append(np.std(trainFeatures[index]))  
    if(trainDigits[index]=="1.0"):
        colors.append("b")
    else:
        colors.append("r")
##normalization     
def normalize(lists):
    norm = [i * 2/(max(lists)-min(lists))+1-2*max(lists)/(max(lists)-min(lists)) for i in lists]
    return norm;
##normalize two features 
Xnorm = normalize(X)
Ynorm = normalize(Y)
Xnorm = np.asarray(Xnorm)
Ynorm = np.asarray(Ynorm)
simpleTrain = np.column_stack((Xnorm,Ynorm))

mp.scatter(Xnorm,Ynorm,c=colors, s=3)


# In[7]:


np.var(Xnorm)+ np.var(Ynorm)


# In[8]:


var = []
for row in range(len(trainFeatures)):
    var.append(np.var(trainFeatures[row]))
var


# In[6]:


model = KNeighborsClassifier(n_neighbors=1)
model.fit(simpleTrain,trainDigits)
xPred = []
yPred = []
cPred = []
for xP in range(-100,100):
    xP = xP/100.0
    for yP in range(-100,100):
        yP = yP/100.0
        xPred.append(xP)
        yPred.append(yP)
        if(model.predict([[xP,yP]])=="1.0"):
            cPred.append("b")
        else:
            cPred.append("r")

mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.2)
show()

