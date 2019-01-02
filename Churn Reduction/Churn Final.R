rm(list=ls())
setwd("D:/PROJECT")
train=read.csv("Train_data.csv",header=T)
test=read.csv("Test_data.csv",header=T)
train=train[,-c(1,2,3,4)]
numeric_index=sapply(train,is.numeric)
numeric_data=train[,numeric_index]
cnames=colnames(numeric_data)
cnames
library(ggplot2)
for(i in 1:length(cnames))
{assign(paste0("gn", i),ggplot(aes_string(y = (cnames[i]),x = "Churn")
                               ,data = subset(train))+ stat_boxplot(geom = "errorbar", width = 0.5)+
          geom_boxplot(outlier.colour="RED",
                       fill ="grey", outlier.shape = 18, outlier.size = 1, notch = FALSE)+theme(legend.position = 'bottom')
        +
          labs(y = cnames[i],x = 'Churn')+
          ggtitle(paste("Box plot of Churn for", cnames[i])))}

library(gridExtra)
gridExtra::grid.arrange(gn1,gn2,gn3,ncol = 3)
gridExtra::grid.arrange(gn4,gn5,gn6,ncol = 3)
gridExtra::grid.arrange(gn7,gn8,gn9,ncol = 3)
gridExtra::grid.arrange(gn10,gn11,gn12,ncol = 3)
gridExtra::grid.arrange(gn13,gn14,ncol = 2)
library(DMwR)
for(i in cnames)
{val=train[,i][train[,i]%in% boxplot.stats(train[,i])$out]
train[,i][train[,i]%in%val]=NA}
train=knnImputation(train,k=3)
library(corrgram)
corrgram(train[,numeric_index],order = F,upper.panel = panel.pie,
         main="correlation plot")
colnames(train)
train=train[,-c(4,7,10,13)]
test=test[,-c(1,2,3,4)]
test=test[,-c(4,7,10,13)]
factor_index=sapply(train,is.factor)
factor_data=train[,factor_index]
for (i in 1:2)
{
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$Churn,factor_data[,i])))
}
library(randomForest)
library(ROSE)
library(caret)
rose=ROSE(Churn~.,data=train,N=3500,seed=1321)$data
rf=randomForest(Churn~.,rose,importance=TRUE,ntree=500)
rf
confusionMatrix(predict(rf,test),test$Churn)
varImpPlot(rf)