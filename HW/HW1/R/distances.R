library(caret)
library(e1071)
library(dplyr)
library(ggplot2)
library(genefilter)
library(class)

# Read data
setwd("~/Documents/Courses/CS412/HW/HW1/R")
#setwd("~/Documents/WW/CS412/HW1/R")
data = read.csv("data.csv",sep=" ",header=FALSE,col.names=append("Digit",seq(1,257,by=1)))
data$X257 = NULL

#selecting 
data = filter(data,Digit==1|Digit==5)
#force R to treat 1,5 as categories, not numerics
data$Digit = as.factor(data$Digit)

#Partitioning the data
set.seed(100)
index = createDataPartition(data$Digit, p = 0.2, list = F )
train = data[index,]
test = data[-index,]
# feature1: means of row intensities of train data
TrainMeanIntensity <- rowMeans(train[,-1])                 
# feature2: standard deviation of row intensities of train data
TrainSDIntensity <- rowSds(as.matrix(train[,-1]))
TrainwithFeatures <- cbind(train, TrainMeanIntensity, TrainSDIntensity)
TrainwithFeatures <- TrainwithFeatures[,-(2:257)]
#test
TestMeanIntensity <- rowMeans(test[,-1])                 
TestSDIntensity <- rowSds(as.matrix(test[,-1]))
TestwithFeatures <- cbind(test, TestMeanIntensity, TestSDIntensity)
TestwithFeatures <- TestwithFeatures[,-(2:257)]
euclideanDist <- function(a,b){
  d = 0
    for(i in c(1:(length(a)-1) ))
    {
      d = d + (a[[i]]-b[[i]])^2
    }
    d = sqrt(d)
    return(d)
}



knn_predict <- function(test_data, train_data, k_value){
  pred <- c()  #empty pred vector 
  #LOOP-1
  for(i in c(1:nrow(test_data))){   #looping over each record of test data
    eu_dist =c()          #eu_dist & eu_char empty  vector
    eu_char = c()
    good = 0              #good & bad variable initialization with 0 value
    bad = 0
    
    #LOOP-2-looping over train data 
    for(j in c(1:nrow(train_data))){
      
      #adding euclidean distance b/w test data point and train data to eu_dist vector
      eu_dist <- c(eu_dist, euclideanDist(test_data[i,], train_data[j,]))
      
      #adding class variable of training data in eu_char
      eu_char <- c(eu_char, as.character(train_data[j,][[6]]))
    }
    
    eu <- data.frame(eu_char, eu_dist) #eu dataframe created with eu_char & eu_dist columns
    
    eu <- eu[order(eu$eu_dist),]       #sorting eu dataframe to gettop K neighbors
    eu <- eu[1:k_value,]               #eu dataframe with top K neighbors
    
    #Loop 3: loops over eu and counts classes of neibhors.
    for(k in c(1:nrow(eu))){
      if(as.character(eu[k,"eu_char"]) == "g"){
        good = good + 1
      }
      else
        bad = bad + 1
    }
    
    # Compares the no. of neighbors with class label good or bad
    if(good > bad){          #if majority of neighbors are good then put "g" in pred vector
      
      pred <- c(pred, "g")
    }
    else if(good < bad){
      #if majority of neighbors are bad then put "b" in pred vector
      pred <- c(pred, "b")
    }
    
  }
  return(pred) #return pred vector
}

knn_predict(TrainwithFeatures,TestwithFeatures,1)
