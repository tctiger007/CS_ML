import numpy as np
import matplotlib.pyplot as mp
from pylab import show
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
data = np.loadtxt("/Users/martinbrennan/Documents/WW/CS412/HW/HW1/data.csv")
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
#polynomial kernel with degree 3
kPCA = KernelPCA(n_components=2, kernel='poly', degree = 3)
kPCA_transform = kPCA.fit_transform(trainFeatures)
#trainFeatures_transformed.shape   (312, 2)

##Figure 2.1
mp.figure()
mp.title("Kernel PCA")
colors = []
for index in range(len(trainFeatures)):
    if(trainDigits[index]=="1.0"):
        colors.append("b")
    else:
        colors.append("r")
mp.scatter(kPCA_transform[:,0], kPCA_transform[:,1],c=colors)
mp.xlabel("1st principal component")
mp.ylabel("2nd principal component")
##############################################################################
explained_variance = np.var(kPCA_transform, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)

################ reuse the HW1 code for Q2 ################
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
    norm = [i * 2/(max(lists)-min(lists))+
            1-2*max(lists)/(max(lists)-min(lists)) for i in lists]
    return norm;
##normalize two features 
Xnorm = normalize(X)
Ynorm = normalize(Y)
Xnorm = np.asarray(Xnorm)
Ynorm = np.asarray(Ynorm)
simpleTrain = np.column_stack((Xnorm,Ynorm))

ls_c1 = LogisticRegression(C = 0.01).fit(simpleTrain, trainDigits)

ls_c2 = LogisticRegression(C = 2).fit(simpleTrain, trainDigits)


