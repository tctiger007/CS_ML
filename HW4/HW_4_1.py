#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import matplotlib.pyplot as mp
from pylab import show
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import math


# In[2]:


#data = np.loadtxt("/Users/martinbrennan/Documents/WW/CS/CS412/HW/HW1/data.csv")
data = np.loadtxt("/Users/wangfei/Documents/Courses/CS/CS412/HW/HW1/data.csv")
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


# In[3]:


## Use the two features that you created for your 2D graph in HW1 
def ExtractFeat(dataset,label):
    X = []   ##mean
    Y = []   ##std
    colors = []
    for index in range(len(dataset)):
        X.append(np.mean(dataset[index]))
        Y.append(np.std(dataset[index]))  
        if(label[index]=="1.0"):
            colors.append("b")
        else:
            colors.append("r")
    return [X, Y, colors];
    
##normalization     
def normalize(lists):
    norm = [i * 2/(max(lists)-min(lists))+
            1-2*max(lists)/(max(lists)-min(lists)) for i in lists]
    return norm;
##normalize two features
###training features
Xnorm = normalize(ExtractFeat(trainFeatures,trainDigits)[0])
Ynorm = normalize(ExtractFeat(trainFeatures,trainDigits)[1])
Xnorm = np.asarray(Xnorm)
Ynorm = np.asarray(Ynorm)
simpleTrain = np.column_stack((Xnorm,Ynorm))
###testing features
Xnorm_test = normalize(ExtractFeat(testFeatures,testDigits)[0])
Ynorm_test = normalize(ExtractFeat(testFeatures,testDigits)[1])
Xnorm_test = np.asarray(Xnorm_test)
Ynorm_test = np.asarray(Ynorm_test)
simpleTest = np.column_stack((Xnorm_test,Ynorm_test))


# In[4]:


xPred = []
yPred = []
for xP in range(-100,100):
    xP = xP/100
    for yP in range(-100,100):
        yP = yP/100
        xPred.append(xP)
        yPred.append(yP)

coordinate = list(zip(xPred,yPred))

def pred(model):
    preds = model.predict(coordinate)
    cPred = []
    for i in range(len(coordinate)):
        if(preds[i] == "1.0"):
            cPred.append("b")
        else:
            cPred.append("r")
    return cPred;            


# In[5]:


img_path = '/Users/wangfei/Documents/Courses/CS/CS412/HW/HW4/Figures/'
#img_path = '/Users/martinbrennan/Documents/WW/CS/CS412/HW/HW4/Figures/'
#define plot function: for plotting decision boundary figures and CV_error figures 
colors = []
for index in range(len(trainFeatures)):
    if(trainDigits[index]=="1.0"):
        colors.append("b")
    else:
        colors.append("r")
def plot(grid1,grid2,plotNum,fileName,plotType,model,
         subtitle,ylimL,ylimU,xlab,ylab,var=None,xscale='linear'):   
    fig = mp.figure()
    if(plotNum==1):
        mp.title(fileName)
        if(plotType==1):  ### decision boundary 
            mp.scatter(Xnorm, Ynorm, c=colors, s=3)
            mp.scatter(xPred, yPred, s=3, c=pred(model),alpha=.2)
        elif(plotType==2): ### CV_error
            mp.errorbar(var, model[0], marker='s', yerr=model[1],fmt='o',
                        markersize=2, capsize=1.5, elinewidth=1)
            mp.xscale(xscale)
        else:
            print('Plot Type not defined.')
        mp.ylim(ylimL,ylimU)
        mp.xlabel(xlab)
        mp.ylabel(ylab)
        mp.savefig(img_path + fileName, dpi = 300)
    else:
        for i in range(1,(plotNum+1)):
            ax = fig.add_subplot(grid1,grid2,i)
            if(plotType==1):
                ax.scatter(Xnorm, Ynorm, c=colors, s=3)
                ax.scatter(xPred, yPred, s=3, c=pred(model[i-1]),alpha=.04)
            elif(plotType==2):
                ax.errorbar(var, model[i-1][0], marker='s', yerr=model[i-1][1],fmt='o',
                            markersize=2, capsize=1.5, elinewidth=1)
                ax.set_xscale(xscale)
            else:
                print('Plot Type not defined.')
            ax.set_title(subtitle[i-1])
            ax.set_ylim(ylimL,ylimU)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            mp.subplots_adjust(top=0.92, bottom=0.12, left=0.11, right=0.94, 
                               hspace=0.60, wspace=0.45)
            mp.savefig(img_path + fileName, dpi = 300)
    return show(); 


# In[6]:


maxLeafNode_1 = np.array([5, 10, 15, 20, 30, 40, 50, 75, 100, 200, 500, 1000, 10000])
minImpurityDecrease = np.array([0,0.00001,0.0001,0.001,0.01,0.1,0.3,0.5,0.7,0.9,1])
#minImpuritySplit = np.array([])
#maxLeafNode_2 = np.array([10, 100, 1000])
nEstimators_1 = np.array([5,10,15,20,30,40,50,75,100,200,500,1000,10000])
maxFeatures = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
nEstimators_2 = np.array([1,10,100,1000])
maxDepth = np.array([1, 10, 100])
nEstimators_3 = np.array([1,5,10,100,1000,10000])

def cv_err(type,var=None,maxLeafNode_2=None,nEstimators_2=None,maxDepth=None): 
    err_bar = []
    err_mean = []
    if(type=='decision_tree'):
        if(var=='max_leaf_nodes'):
            for index in range(len(maxLeafNode_1)):
                clf = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=maxLeafNode_1[index])
                acc = cross_val_score(clf, simpleTrain, trainDigits, cv = 10)
                err = 1 - acc
                err_mean.append(err.mean())
                err_bar.append(1.96 * err.std())
                cv = [err_mean,err_bar]
                upper_bound = np.asarray(cv[0])+np.asarray(cv[1])
                min_upper_bound = np.min(upper_bound)
                indx = np.argmin(upper_bound)
                opt = maxLeafNode_1[indx]
        elif(var=='min_impurity_decrease'):
            for index in range(len(minImpurityDecrease)):
                clf = DecisionTreeClassifier(criterion='entropy',min_impurity_decrease=minImpurityDecrease[index])
                acc = cross_val_score(clf, simpleTrain, trainDigits, cv = 10)
                err = 1 - acc
                err_mean.append(err.mean())
                err_bar.append(1.96 * err.std())
                cv = [err_mean,err_bar]
                upper_bound = np.asarray(cv[0])+np.asarray(cv[1])
                min_upper_bound = np.min(upper_bound)
                indx = np.argmin(upper_bound)
                opt = minImpurityDecrease[indx]
        else:
            print('Invalid var')
    elif(type=='random_forest'):
        if(var=='max_leaf_nodes'):
            for index in range(len(nEstimators_1)):
                clf = RandomForestClassifier(criterion='entropy',max_leaf_nodes=maxLeafNode_2,
                                             n_estimators=nEstimators_1[index])
                acc = cross_val_score(clf, simpleTrain, trainDigits, cv = 10)
                err = 1 - acc
                err_mean.append(err.mean())
                err_bar.append(1.96 * err.std())
                cv = [err_mean,err_bar]
                upper_bound = np.asarray(cv[0])+np.asarray(cv[1])
                min_upper_bound = np.min(upper_bound)
                indx = np.argmin(upper_bound)
                opt = nEstimators_1[indx]
        elif(var=='graduate_question'):
            for index in range(len(maxFeatures)):
                clf = RandomForestClassifier(criterion='entropy',
                                             max_leaf_nodes=maxLeafNode_2,
                                             n_estimators=nEstimators_2,
                                             max_features=maxFeatures[index])
                acc = cross_val_score(clf, simpleTrain, trainDigits, cv = 10)
                err = 1 - acc
                err_mean.append(err.mean())
                err_bar.append(1.96 * err.std())
                cv = [err_mean,err_bar]
                upper_bound = np.asarray(cv[0])+np.asarray(cv[1])
                min_upper_bound = np.min(upper_bound)
                indx = np.argmin(upper_bound)
                opt = maxFeatures[indx]
        else:
            print('Invalid var')
    elif(type=='AdaBoost'):
        for index in range(len(nEstimators_3)):
            clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',
                                                                           max_depth=maxDepth),
                                     n_estimators=nEstimators_3[index])
            acc = cross_val_score(clf, simpleTrain, trainDigits, cv = 10)
            err = 1 - acc
            err_mean.append(err.mean())
            err_bar.append(1.96 * err.std())
            cv = [err_mean,err_bar]
            upper_bound = np.asarray(cv[0])+np.asarray(cv[1])
            min_upper_bound = np.min(upper_bound)
            indx = np.argmin(upper_bound)
            opt = nEstimators_3[indx]
    else:
            print('Invalid type')
    return [err_mean,err_bar,opt,indx];


# In[7]:


#Figure 4_1 part a)
plot(grid1=1,grid2=1,plotNum=1,fileName='Figure4_1.png',plotType=2,
     model=cv_err('decision_tree','max_leaf_nodes'),subtitle=(''),ylimL=-0.1,
     ylimU=0.4,xlab='Max leaf nodes', ylab='CV Error',var = maxLeafNode_1,xscale='log')


# In[8]:


#c) lowest cv error from part a)
optModelPartC = DecisionTreeClassifier(criterion='entropy',
                                       max_leaf_nodes=cv_err('decision_tree','max_leaf_nodes')[2])
plot(grid1=1,grid2=1,plotNum=1,fileName='Figure4_2.png',plotType=1,
     model=optModelPartC.fit(simpleTrain, trainDigits),subtitle=(''),
     ylimL=-1,ylimU=1,xlab='Mean Intensity',ylab='Intensity SD')


# In[9]:


#Graduate student question part d)
plot(grid1=1,grid2=1,plotNum=1,fileName='Figure4_d.png',plotType=2,
     model=cv_err('decision_tree','min_impurity_decrease'),subtitle=(''),ylimL=-0.1,
     ylimU=0.4,xlab='Min impurity decrease', ylab='CV Error',
     var = minImpurityDecrease,xscale='log')
cv_err('decision_tree','min_impurity_decrease')


# In[10]:


#Figure 4_3, 4.4, 4.5 part e)
modelPartE1 = cv_err('random_forest','max_leaf_nodes',maxLeafNode_2=10)
modelPartE2 = cv_err('random_forest','max_leaf_nodes',maxLeafNode_2=100)
modelPartE3 = cv_err('random_forest','max_leaf_nodes',maxLeafNode_2=1000)
plot(grid1=1,grid2=1,plotNum=1,fileName='Figure4_3.png',plotType=2,
     model=modelPartE1,subtitle=('max leaf nodes = 10'),ylimL=-0.1,ylimU=0.4,
     xlab='n Estimators', ylab='CV Error',var = nEstimators_1,xscale='log')
plot(grid1=1,grid2=1,plotNum=1,fileName='Figure4_4.png',plotType=2,
     model=modelPartE2,subtitle=('max leaf nodes = 100'),ylimL=-0.1,ylimU=0.4,
     xlab='n Estimators', ylab='CV Error',var = nEstimators_1,xscale='log')
plot(grid1=1,grid2=1,plotNum=1,fileName='Figure4_5.png',plotType=2,
     model=modelPartE3,subtitle=('max leaf nodes = 1000'),ylimL=-0.1,ylimU=0.4,
     xlab='n Estimators', ylab='CV Error',var = nEstimators_1,xscale='log')


# In[20]:


[modelPartE1, modelPartE2, modelPartE3]
[np.var(modelPartE1[0]), np.var(modelPartE2[0]), np.var(modelPartE3[0])]


# In[12]:



[0.045181451612903226+0.06535505429879578,
 0.02915322580645161+0.061036035227182824,
 0.03237903225806451+0.057988138745537404]


# In[14]:


#h) decision boundary for random forest 
optModelPartH = RandomForestClassifier(criterion='entropy',max_leaf_nodes=100,
                                       n_estimators=40)
plot(grid1=1,grid2=1,plotNum=1,fileName='Figure4_6.png',plotType=1,
     model=optModelPartH.fit(simpleTrain, trainDigits),subtitle=(''),
     ylimL=-1,ylimU=1,xlab='Mean Intensity',ylab='Intensity SD')


# In[16]:


#graduate student question part i 
modelPartI10_1 = cv_err('random_forest','graduate_question',maxLeafNode_2=10,nEstimators_2=1)
modelPartI10_10 = cv_err('random_forest','graduate_question',maxLeafNode_2=10,nEstimators_2=10)
modelPartI10_100 = cv_err('random_forest','graduate_question',maxLeafNode_2=10,nEstimators_2=100)
modelPartI10_1000 = cv_err('random_forest','graduate_question',maxLeafNode_2=10,nEstimators_2=1000)
modelPartI100_1 = cv_err('random_forest','graduate_question',maxLeafNode_2=100,nEstimators_2=1)
modelPartI100_10 = cv_err('random_forest','graduate_question',maxLeafNode_2=100,nEstimators_2=10)
modelPartI100_100 = cv_err('random_forest','graduate_question',maxLeafNode_2=100,nEstimators_2=100)
modelPartI100_1000 = cv_err('random_forest','graduate_question',maxLeafNode_2=100,nEstimators_2=1000)
modelPartI1000_1 = cv_err('random_forest','graduate_question',maxLeafNode_2=1000,nEstimators_2=1)
modelPartI1000_10 = cv_err('random_forest','graduate_question',maxLeafNode_2=1000,nEstimators_2=10)
modelPartI1000_100 = cv_err('random_forest','graduate_question',maxLeafNode_2=1000,nEstimators_2=100)
modelPartI1000_1000 = cv_err('random_forest','graduate_question',maxLeafNode_2=1000,nEstimators_2=1000)


# In[17]:


CV_error_PartI = np.array([modelPartI10_1[0],modelPartI10_10[0],modelPartI10_100[0],modelPartI10_1000[0],
        modelPartI100_1[0],modelPartI100_10[0],modelPartI100_100[0],modelPartI100_1000[0],
        modelPartI1000_1[0],modelPartI1000_10[0],modelPartI1000_100[0],modelPartI1000_1000[0]])
np.savetxt("CV_error_PartI.csv",CV_error_PartI)


# In[32]:


#adaboost
modelPartJ = cv_err('AdaBoost',maxDepth=1)
modelPartK = cv_err('AdaBoost',maxDepth=10)
modelPartL = cv_err('AdaBoost',maxDepth=1000)


# In[33]:


#part j 
plot(grid1=1,grid2=1,plotNum=1,fileName='Figure4_7.png',plotType=2,
     model=modelPartJ,subtitle=('max depth = 1'),ylimL=-0.1,ylimU=0.4,
     xlab='n Estimators', ylab='CV Error',var = nEstimators_3,xscale='log')
#part k
plot(grid1=1,grid2=1,plotNum=1,fileName='Figure4_8.png',plotType=2,
     model=modelPartK,subtitle=('max depth = 10'),ylimL=-0.1,ylimU=0.4,
     xlab='n Estimators', ylab='CV Error',var = nEstimators_3,xscale='log')
#part l
plot(grid1=1,grid2=1,plotNum=1,fileName='Figure4_9.png',plotType=2,
     model=modelPartL,subtitle=('max depth = 1000'),ylimL=-0.1,ylimU=0.4,
     xlab='n Estimators', ylab='CV Error',var = nEstimators_3,xscale='log')


# In[17]:


#part m
optModelPartM = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',
                                                                           max_depth=1),
                                     n_estimators=100)
plot(grid1=1,grid2=1,plotNum=1,fileName='Figure4_10.png',plotType=1,
     model=optModelPartM.fit(simpleTrain, trainDigits),subtitle=(''),
     ylimL=-1,ylimU=1,xlab='Mean Intensity',ylab='Intensity SD')


# In[104]:


####Part 2 
SVM_clf = SVC(kernel='poly',C = 38.88155180308085,degree = 2,gamma='auto')
NN_clf = MLPClassifier(hidden_layer_sizes=(50,2),activation = "relu", epsilon=0.001, 
                       max_iter=10000,alpha=0, solver = "adam")
def bound(model):
    fit = model.fit(simpleTrain,trainDigits)
    error = 1- fit.score(simpleTest,testDigits)
    Markov_75 = error/0.25
    Markov_95 = error/0.05
    Markov_99 = error/0.01
    n = len(testDigits)
    var = error * (1-error)/n
    cheby_75 = error + math.sqrt(var/0.25)
    cheby_95 = error + math.sqrt(var/0.05)
    cheby_99 = error + math.sqrt(var/0.01)
    Hoeff_75 = error + math.sqrt(1/2/n*math.log(2/0.25))
    Hoeff_95 = error + math.sqrt(1/2/n*math.log(2/0.05))
    Hoeff_99 = error + math.sqrt(1/2/n*math.log(2/0.01))
    return np.array([Markov_75,Markov_95,Markov_99,
                    cheby_75,cheby_95,cheby_99,
                    Hoeff_75,Hoeff_95,Hoeff_99])

