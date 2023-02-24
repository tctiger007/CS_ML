library(caret)
library(e1071)
library(dplyr)
library(ggplot2)
library(genefilter)
library(knnGarden)
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
TrainSDIntensity <- rowSds(train[,-1])
TrainwithFeatures <- cbind(train, TrainMeanIntensity, TrainSDIntensity)
TrainwithFeatures <- TrainwithFeatures[,-(2:257)]
#Figure 1.1
#png("Figure1.1.png", units = "in", width = 8, height = 6, res = 600)
graph1.1 <- ggplot(TrainwithFeatures,aes(x=TrainMeanIntensity,y=TrainSDIntensity))+ 
  geom_point(aes(color=Digit)) + 
  labs (x = "Mean Intensity", y = "Standard Deviation of Intensity") + 
  xlim(c(-1,0)) + ylim(c(0.4, 1)) + 
  theme_bw() + theme (panel.grid.major = element_blank(),
                      panel.grid.minor = element_blank())
graph1.1
#dev.off()

#test
TestMeanIntensity <- rowMeans(test[,-1])                 
TestSDIntensity <- rowSds(test[,-1])
TestwithFeatures <- cbind(test, TestMeanIntensity, TestSDIntensity)
TestwithFeatures <- TestwithFeatures[,-(2:257)]
#create lattice; label the graph
x2 <- rep(0,6161)
for (i in 1:61){
  seq <- seq(0.4,1,0.01)
  x2[(1+(i-1)*101):(101+(i-1)*101)] <- rep(seq[i],101)
}
lattice <- matrix(c(rep(seq(-1,0,0.01),61), x2), nrow = 6161, ncol = 2)
#1 nearest neighbor; euclidean dist
model1 <- knn(TrainwithFeatures[,-1], lattice, TrainwithFeatures$Digit, k=1, prob=TRUE)
prob <- attr(model1, "prob")
prob <- ifelse(model1=="1", prob, 1-prob)
#lattice coordinates
x <- seq(-1, 0, 0.01)
y <- seq(0.4, 1, 0.01)
prob1 <- matrix(prob, length(x), length(y))
#Figure 1.2 Figure 1.4
#png("Figure1_2_v2.png", units = "in", width = 8, height = 6, res = 300)
par(mar=c(4,4,4,5), xpd=TRUE)
contour(x, y, prob1, levels=.5, labels = "", xlab = "Mean Intensity", ylab = "Standard Deviation of Intensity", 
        main= "1-nearest neighbour", axes=TRUE, labcex = 0.5)
points(TrainwithFeatures$TrainMeanIntensity,TrainwithFeatures$TrainSDIntensity, type = "p",
       pch = 20, col=ifelse(TrainwithFeatures$Digit==1, "#F8766D", "#00BFC4"))
gd <- expand.grid(x=x, y=y)
legend("topright", inset=c(-0.14,0.4), legend=c("1","5"), pch=c(16,16), 
       col = c("#F8766D", "#00BFC4"), title="Digit", bty = "n")
points(gd, pch=".", cex=1.2, col=ifelse(prob1>0.5, "#F8766D", "#00BFC4"))
#dev.off()     

#k-nearest neighbor
#Set the level of cross-validation
trControl <- trainControl(method  = "cv",
                          number  = 10)
model.all <- train(Digit~., #these are the predictive variables
                   data = train, 
                   method = "knn",
                   trControl = trControl,
                   tuneGrid = expand.grid(k = seq(1,49,2))) #modeling 1s and 5s for 256 dimensions
#Figure 1.3
#png("Figure1_3.png", units = "in", width = 8, height = 6, res = 600)
cv.256d.df <- cbind(model.all$results$k, 1-model.all$results$Accuracy,
                    model.all$results$AccuracySD)
colnames(cv.256d.df) <- c("k", "Ecv", "SD")
cv.256d.df <- as.data.frame(cv.256d.df)
ggplot(cv.256d.df, aes(x=k, y=Ecv)) + 
  geom_line(color = "dodgerblue") + 
  geom_point(color = "dodgerblue") + 
  ggtitle("Cross validation errors for k neighbors, 256-dimensioned") +
  theme_bw() +
  theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))
#dev.off() 

#95% CI 
c(cv.256d.df$Ecv+cv.256d.df$SD)








model.twofeature <- train(Digit~TrainMeanIntensity+TrainSDIntensity , #these are the predictive variables
                          data = TrainwithFeatures, 
                          method = "knn",
                          trControl = trControl,
                          tuneGrid   = expand.grid(k = seq(1,49,2))) #modeling 1s and 5s for 256 dimensions
model.twofeature.error <- 1 - model.twofeature$results$Accuracy[1]

#Figure 1.4
#png("Figure1_4.png", units = "in", width = 8, height = 6, res = 600)
cv.2d.df <- cbind(model.twofeature$results$k, 1-model.twofeature$results$Accuracy)
colnames(cv.2d.df) <- c("k","Ecv")
cv.2d.df <- as.data.frame(cv.2d.df)
ggplot(cv.2d.df, aes(x=k, y=Ecv)) + 
  geom_line(color = "dodgerblue") + 
  geom_point(color = "dodgerblue") + 
  ggtitle("Cross validation errors for k neighbors, 2-dimensioned") +
  theme_bw() +
  theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))
#dev.off()    

cv.compare <- cbind(model.twofeature$results$k,1-model.twofeature$results$Accuracy,
                    1-model.all$results$Accuracy)
colnames(cv.compare) <- c("k","2-dimension","256-dimension")




# b) (EC) Manhattan distance - 2 dimensions
# c) (EC) Chebyshev distance - 2 dimensions
# d) Euclidean distance - 256 dimensions
# e) (EC) Manhattan distance - 256 dimensions 
# f) (EC) Chebyshev distance - 256 dimensions





