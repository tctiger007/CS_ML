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
TrainMeanIntensity <- 2/(max(TrainMeanIntensity)-min(TrainMeanIntensity))*
  TrainMeanIntensity+1-2*max(TrainMeanIntensity)/(max(TrainMeanIntensity)-min(TrainMeanIntensity))
# feature2: standard deviation of row intensities of train data
TrainSDIntensity <- rowSds(as.matrix(train[,-1]))
TrainSDIntensity <- 2/(max(TrainSDIntensity)-min(TrainSDIntensity))*
  TrainSDIntensity+1-2*max(TrainSDIntensity)/(max(TrainSDIntensity)-min(TrainSDIntensity))
TrainwithFeatures <- cbind(train, TrainMeanIntensity, TrainSDIntensity)
TrainwithFeatures <- TrainwithFeatures[,-(2:257)]
#Figure 1.1
#png("Figure1.1.png", units = "in", width = 8, height = 6, res = 600)
graph1.1 <- ggplot(TrainwithFeatures,aes(x=TrainMeanIntensity,y=TrainSDIntensity))+ 
  geom_point(aes(color=Digit)) + 
  labs (x = "Mean Intensity", y = "Standard Deviation of Intensity") + 
  theme_bw() + theme (panel.grid.major = element_blank(),
                      panel.grid.minor = element_blank())
graph1.1
#dev.off()

#test
TestMeanIntensity <- rowMeans(test[,-1])                 
TestSDIntensity <- rowSds(as.matrix(test[,-1]))
TestwithFeatures <- cbind(test, TestMeanIntensity, TestSDIntensity)
TestwithFeatures <- TestwithFeatures[,-(2:257)]

#create lattice; label the graph  [-1,1] [-1,1]
x2.2 <- rep(0,201*201)
for (i in 1:201){
  seq <- seq(-1,1,0.01)
  x2.2[(1+(i-1)*201):(201+(i-1)*201)] <- rep(seq[i],201)
}
lattice <- matrix(c(rep(seq(-1,1,0.01),201), x2.2), nrow = 201*201, ncol = 2)

#1 nearest neighbor; euclidean dist
model1.2 <- knn(TrainwithFeatures[,-1], lattice, TrainwithFeatures$Digit, k=1, prob=TRUE)
prob.2 <- attr(model1.2, "prob")
prob.2 <- ifelse(model1.2=="1", prob.2, 1-prob.2)
#lattice coordinates
x.2 <- seq(-1, 1, 0.01)
y.2 <- seq(-1, 1, 0.01)
prob1.2 <- matrix(prob.2, length(x.2), length(y.2))
#png("Figure1_2_v1.png", units = "in", width = 8, height = 6, res = 300)
par(mar=c(4,4,4,5), xpd=TRUE)
contour(x.2, y.2, prob1.2, levels=.5, labels = "", xlab = "Mean Intensity", ylab = "Standard Deviation of Intensity", 
        main= "1-nearest neighbour", axes=TRUE, labcex = 0.5)
points(TrainwithFeatures$TrainMeanIntensity,TrainwithFeatures$TrainSDIntensity, type = "p",
       pch = 20, col=ifelse(TrainwithFeatures$Digit==1, "#F8766D", "#00BFC4"))
gd.2 <- expand.grid(x=x.2, y=y.2)
legend("topright", inset=c(-0.14,0.4), legend=c("1","5"), pch=c(16,16), 
       col = c("#F8766D", "#00BFC4"), title="Digit", bty = "n")
points(gd.2, pch=".", cex=1.2, col=ifelse(prob1.2>0.5, "#F8766D", "#00BFC4"))
#dev.off()

#k-nearest neighbor
#Set the level of cross-validation
trControl <- trainControl(method  = "cv",
                          number  = 10)
model.all <- train(Digit~., data = train, 
                   method = "knn", trControl = trControl,
                   tuneGrid = expand.grid(k = seq(1,49,2))) 
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
CI.lower <- c(round(cv.256d.df$Ecv - 2*cv.256d.df$SD, digits = 3))
CI.upper <- c(round(cv.256d.df$Ecv + 2*cv.256d.df$SD, digits = 3))
cbind(seq(1,25,1), CI.lower,CI.upper)

#2d model
model.twofeature <- train(Digit~TrainMeanIntensity+TrainSDIntensity, 
                   data = TrainwithFeatures, method = "knn",
                   trControl = trControl, 
                   tuneGrid = expand.grid(k = seq(1,49,2))) 

cv.2d.df <- cbind(model.twofeature$results$k, 1-model.twofeature$results$Accuracy)
colnames(cv.2d.df) <- c("k","Ecv")
cv.2d.df <- as.data.frame(cv.2d.df)
ggplot(cv.2d.df, aes(x=k, y=Ecv)) + 
  geom_line(color = "dodgerblue") + 
  geom_point(color = "dodgerblue") + 
  ggtitle("Cross validation errors for k neighbors, 2-dimensioned") +
  theme_bw() +
  theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))

cv.2d.df <- cbind(model.twofeature$results$k, 1-model.all$results$Accuracy,
                    model.all$results$AccuracySD)
colnames(cv.256d.df) <- c("k", "Ecv", "SD")
cv.256d.df <- as.data.frame(cv.256d.df)
ggplot(cv.256d.df, aes(x=k, y=Ecv)) + 
  geom_line(color = "dodgerblue") + 
  geom_point(color = "dodgerblue") + 
  ggtitle("Cross validation errors for k neighbors, 256-dimensioned") +
  theme_bw() +
  theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))

cv.compare <- cbind(model.twofeature$results$k,1-model.twofeature$results$Accuracy,
                    1-model.all$results$Accuracy)
colnames(cv.compare) <- c("k","2-dimension","256-dimension")

#####extra credit 
library(matrixStats)
library(FactoMineR)
library(factoextra)
setwd("~/Documents/Courses/BIOE530/Final Project/HAPT Data Set")

# Load data
X_train = read.table("Train/X_train.txt")
Y_train = unlist(read.table("Train/Y_train.txt"))
subj_id_train = unlist(read.table("Train/subject_id_train.txt"))

X_test = read.table("Test/X_test.txt")
Y_test = unlist(read.table("Test/Y_test.txt"))
subj_id_test = unlist(read.table("Test/subject_id_test.txt"))

activityLabels = read.table("activity_labels.txt")$V2
featureLabels = unlist(read.table("features.txt"))

colnames(X_train) = featureLabels
colnames(X_test) = featureLabels

## PCA
res.pca = PCA(X_train, ncp=30, graph = FALSE)
which( colnames(X_train)=="fBodyGyro-SMA-1" )  #439
which( colnames(X_train)=="fBodyAcc-SMA-1" )  #281

df <- cbind(Y_train,X_train$`fBodyAcc-SMA-1`,X_train$`fBodyGyro-SMA-1`)
df <- as.data.frame(df)
colnames(df) <- c("activity", "feature1", "feature2")
df1 <- filter(df, activity == "4"|activity == "5"|activity == "6")
df1$activity <- "Static"
df2 <- filter(df, activity == "1"|activity == "2"|activity == "3")
df2$activity <- "Dynamic"
df3 <- filter(df, activity == "7"|activity == "8"|activity == "9"|
                activity == "10"|activity == "11"|activity == "12")
df3$activity <- "Postural_transition"

df <- rbind(df1,df2,df3)
df <- as.data.frame(df)

#png("Figure1_5.png", units = "in", width = 8, height = 6, res = 600)
ggplot(df,aes(x=feature1,y=feature2))+ 
  geom_point(aes(color=activity)) + 
  labs (x = "fBodyAcc-SMA-1", y = "fBodyGyro-SMA-1") + 
  theme_bw() + theme (panel.grid.major = element_blank(),
                      panel.grid.minor = element_blank())
#dev.off()
model.postual <- train(activity~., data = df, method = "knn", 
                       trControl = trControl, tuneGrid = expand.grid(k = seq(1,100,2))) 

#png("Figure1_6.png", units = "in", width = 8, height = 6, res = 600)
cv.postual <- cbind(model.postual$results$k, (1-model.postual$results$Accuracy))
colnames(cv.postual) <- c("k", "Ecv")
cv.postual <- as.data.frame(cv.postual)
plot(cv.postual$k,cv.postual$Ecv)
ggplot(cv.postual, aes(x=k, y=Ecv)) + 
  geom_line(color = "dodgerblue") + 
  geom_point(color = "dodgerblue") + 
  ggtitle("Cross validation errors for k neighbors, 561 dimensions") +
  theme_bw() +
  theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))
#dev.off() 

# glass dataset 
setwd("~/Documents/Courses/CS412/HW/HW1/R/glass")
glass <- read.table("glass.data", sep=",")
colnames(glass) <- c("ID", "RI", "Na", "Mg", "Al", "Si", "K",
                     "Ca", "Ba", "Fe", "GlassType")
glass.pca = PCA(glass[,2:10], ncp=10, graph = FALSE)
glass.pca$var$contrib
glass$GlassType <- as.factor(glass$GlassType)
glass <- glass[, -1]

for (k in 1:9){feature <- glass[,k]
feature <- as.numeric(feature)
feature <- 2/(max(feature)-min(feature))*feature+1-2*max(feature)/(max(feature)-min(feature))
glass[,k] <- feature
}

png("Figure1_7.png", units = "in", width = 8, height = 6, res = 600)
ggplot(glass, aes(x = Ca, y = RI)) + 
  geom_point(aes(color=GlassType)) + 
  labs (x = "Calcium", y = "Refractive Index") + 
  theme_bw() + theme (panel.grid.major = element_blank(),
                      panel.grid.minor = element_blank())
dev.off()
model.glass <- train(GlassType~., data = glass, 
                          method = "knn", trControl = trControl,
                          tuneGrid = expand.grid(k = seq(1,49,2))) 

cv.glass <- cbind(model.glass$results$k, 1-model.glass$results$Accuracy)
colnames(cv.glass) <- c("k","Ecv")
cv.glass <- as.data.frame(cv.glass)
#png("Figure1_8.png", units = "in", width = 8, height = 6, res = 600)
ggplot(cv.glass, aes(x=k, y=Ecv)) + 
  geom_line(color = "dodgerblue") + 
  geom_point(color = "dodgerblue") + 
  ggtitle("Cross validation errors for k neighbors, all features") +
  theme_bw() +
  theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))
#dev.off()    


