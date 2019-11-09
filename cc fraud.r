library(dplyr)
library(ggplot2)
library(PRROC)
library(discreteRV)
library(caret)
library(car)
library(glmnet)
library(InformationValue)
library(devtools)
# library(ggbiplot)
# library(remotes)
# library(ggfortify)
# library(cluster)
library(ggbiplot)
library(factoextra)
library(data.table)

df <- fread("creditcard.csv")

df <- setDF(df)

head(df)

anyNA(df)

table(df$Class)

dim(df)

sample_size = floor(0.7*nrow(df))
set.seed(123)

picked = sample(seq_len(nrow(df)),size = sample_size)
train =df[picked,]
test =df[-picked,]

dim(train)
table(train$Class)

dim(test)
table(test$Class)

# ?prcomp

# PCA works best with numerical data, exclude the two categorical variables

# PC1 (time) explains 6.3% of total variance.
# PC31 (class) explains 0.1% of total variance

pca <- prcomp(train[,c(2:29)],center = TRUE,scale. = TRUE)

pca2 <- prcomp(train[,c(1,30)],center = TRUE, scale. = TRUE)

pca_all <- prcomp(train[,c(-31)], center = TRUE, scale. = TRUE)

summary(pca)

summary(pca2)

summary(pca_all)

# The center point ($center), scaling ($scale), standard deviation(sdev) of each principal component
# The relationship (correlation or anticorrelation, etc) between the initial variables and the principal components ($rotation)
# The values of each sample in terms of the principal components ($x)

str(pca)

str(pca2)

str(pca_all)

ggbiplot(pca)

ggbiplot(pca2)

ggbiplot(pca, choices = c(1,2))

ggbiplot(pca, labels = rownames(df$Class))

ggbiplot(pca, obs.scale = 1, var.scale = 1,var.axes=FALSE, labels=rownames(df), groups=df$Class,
        ellipse = TRUE, circle = TRUE)


# factoextra
##############

fviz_eig(pca_all)
# Visualize eigenvalues (scree plot). Show the percentage of variances explained by each principal component.

# fviz_pca_ind(pca_all,
#              col.ind = "cos2", # Color by the quality of representation
#              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
#              repel = TRUE     # Avoid text overlapping
#              )
# # Graph of individuals. Individuals with a similar profile are grouped together.

fviz_pca_var(pca_all,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
             )

fviz_pca_var(pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
             )

fviz_pca_var(pca2,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
             )

# Eigenvalues
eig.val <- get_eigenvalue(pca_all)
eig.val
  
# Results for Variables
res.var <- get_pca_var(pca_all)
res.var$coord          # Coordinates
res.var$contrib        # Contributions to the PCs
res.var$cos2           # Quality of representation 
# Results for individuals
res.ind <- get_pca_ind(pca_all)
res.ind$coord          # Coordinates
res.ind$contrib        # Contributions to the PCs
res.ind$cos2           # Quality of representation 

# Eigenvalues
eig.val <- get_eigenvalue(pca)
eig.val
  
# Results for Variables
res.var <- get_pca_var(pca)
res.var$coord          # Coordinates
res.var$contrib        # Contributions to the PCs
res.var$cos2           # Quality of representation 
# Results for individuals
res.ind <- get_pca_ind(pca)
res.ind$coord          # Coordinates
res.ind$contrib        # Contributions to the PCs
res.ind$cos2           # Quality of representation 

# Predict the coordinates of new individuals data. Use the R base function predict():
pred <- predict(pca_all,test)

head(pred)

# # Plot of active individuals
# p <- fviz_pca_ind(pca_all, repel = TRUE)
# # Add supplementary individuals
# fviz_add(p, pred, color ="blue")

# The predicted coordinates of individuals can be manually calculated as follow:

# Center and scale the new individuals data using the center and the scale of the PCA
# Calculate the predicted coordinates by multiplying the scaled values with the eigenvectors (loadings) of the principal components.

# Centering and scaling the supplementary individuals
pred_scaled <- scale(train, 
                    center = pca_all$center,
                    scale = pca_all$scale)
# Coordinates of the individividuals
coord_func <- function(ind, loadings){
  r <- loadings*ind
  apply(r, 2, sum)
}
pca.loadings <- pca_all$rotation
pred <- t(apply(pred_scaled, 1, coord_func, pca.loadings ))
head(pred)



















###################################
## ggfortify

princomp(df,cor=FALSE,score=TRUE)

autoplot(prcomp(df),colour = 'Class', loadings = TRUE)

autoplot(clara(df[2:29],3))









###################################

df_logit <- df[,-which(names(df) %in% c("V2","V5","V7","Amount"))]

set.seed(1)
test_index <- createDataPartition(y = df_logit$Class, times = 1, p = 0.3, list = FALSE)
train <- df_logit[-test_index,]
test <- df_logit[test_index,]

# LOGISTIC REGRESSION

# http://r-statistics.co/Logistic-Regression-With-R.html

logitpredtrain <- glm(Class~.,
                      train, family=binomial(link="logit"))

#######
# cross validate

x = model.matrix(Class ~ ., df_logit)[, -1] #Dropping the intercept column.
y = df_logit$Class

lambdas = NULL
for (i in 1:10)
{
    fit <- cv.glmnet(x,y)
    errors = data.frame(fit$lambda,fit$cvm)
    lambdas <- rbind(lambdas,errors)
}
# take mean cvm for each lambda
lambdas <- aggregate(lambdas[, 2], list(lambdas$fit.lambda), mean)

# select the best one
bestindex = which(lambdas[2]==min(lambdas[2]))
bestlambda = lambdas[bestindex,1]

# and now run glmnet once more with it
fit <- glmnet(x,y,lambda=bestlambda)

fit

fit$beta

fit$lambda

results <- predict(fit, s = bestlambda, type = "coefficients")
choicePred<-rownames(results)[which(results !=0)]
choicePred

bestlambda = predict(fit, s = bestlambda, newx = x)
mean((bestlambda - y)^2)

# AOC

predicted <- predict(logitpredtrain, test, type="response")

summary(logitpredtrain)

AIC(logitpredtrain)
BIC(logitpredtrain)

# check for multicollinearity
as.data.frame(vif(logitpredtrain))

optCutOff <- optimalCutoff(test$Class, predicted)[1] 
misClassError(test$Class, predicted, threshold = optCutOff)
# The lower the misclassification error, the better is your model

plotROC(test$Class, predicted)
#Receiver Operating Characteristics Curve traces the percentage of true positives accurately 
#predicted by a given logit model as the prediction probability cutoff is lowered from 1 to 0. 
#For a good model, as the cutoff is lowered, 
#it should mark more of actual 1’s as positives and lesser of actual 0’s as 1’s. 
#So for a good model, the curve should rise steeply, 
#indicating that the TPR (Y-Axis) increases faster than the FPR (X-Axis) as the cutoff score decreases. 
#Greater the area under the ROC curve, better the predictive ability of the model.


# confusMat <- confusionMatrix(test$Class, predicted, threshold = optCutOff)
# confusMat

# accuracy <- sum(diag(as.matrix(confusMat)))/sum(confusMat)
# print('--Overall Accuracy--')
# accuracy





no_fraud=df_logit[df_logit$Class==0,]
fraud=df_logit[df_logit$Class==1,]

pr <- pr.curve(no_fraud$V1,fraud$V2,curve=T)
pr

plot(pr)

y <- as.data.frame(pr$curve)
ggplot(y,aes(y$V1,y$V2))+geom_path()+ylim(0,1)

pred <- prediction(df$V1,df$Class)

perf <- performance(pred,"tpr","fpr")

plot(perf)

perf1 <- performance(pred,"prec","rec")

plot(perf1)

x <- perf1@x.values[[1]] # recall values
y <- perf1@y.values[[1]] # precision values






