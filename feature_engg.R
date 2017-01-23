library("mlbench")
library("caret")
library("mice")
library("dplyr")

train_df <- read.csv("train.csv")
train_df <- within(train_df,rm("Id"))

test_df <- read.csv("test.csv")
ids <- test_df$Id
test_df <- within(test_df,rm("Id"))

prices <- train_df$SalePrice
train_df <- within(train_df,rm("SalePrice"))

numeric_features <- c("LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "X1stFlrSF", "X2ndFlrSF", "LowQualFinSF","GrLivArea", "BsmtFullBath", "BsmtHalfBath","FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr","TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars","GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch","X3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold","YrSold")

train_df_numeric <- train_df[numeric_features]
train_df_factor <- train_df[!names(train_df) %in% names(train_df_numeric)]

test_df_numeric <- test_df[numeric_features]
test_df_factor <- test_df[!names(test_df) %in% names(test_df_numeric)]
# **** Ordered Categorical to be treated independently ****

# Imputation 
train_plus_test <- rbind(train_df_numeric,test_df_numeric)
col_with_NA <- colnames(train_plus_test)[colSums(is.na(train_plus_test)) > 0]
imputed <- mice::mice(train_plus_test, m = 5, maxit = 10, seed = 500)
imputed_df <- complete(imputed,1) # ***** implement averaging the 5 dfs ******

train_df_numeric <- dplyr::slice(imputed_df,1:1460)
test_df_numeric <- dplyr::slice(imputed_df,1461:2919)

# Feature Scaling
scaleFunc <- function(x) (x - mean(x))/sd(x)
train_df_numeric <- data.frame(lapply(train_df_numeric, scaleFunc))
test_df_numeric <- data.frame(lapply(test_df_numeric, scaleFunc))

# Removing Correlated features
getCorrelated <- function(x) {
  correlationMatrix <- cor(x)
  highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
  return(highlyCorrelated)
}

trainCorrelated <- getCorrelated(train_df_numeric)
toRemove <- trainCorrelated[seq(1,length(trainCorrelated),2)]
train_df_numeric <- train_df_numeric[,-c(toRemove)]

# One hot Encoding
train_df_factor <- mydummies::dummy.data.frame(train_df_factor,names = names(train_df_factor) ,sep = "_", omit.constants=TRUE)
test_df_factor <- mydummies::dummy.data.frame(test_df_factor,names = names(test_df_factor) ,sep = "_", omit.constants=TRUE)
# ******* use (n-1) factor levels *******

# Combine all features
train_df <- cbind(train_df_factor,train_df_numeric)
test_df <- cbind(test_df_factor,test_df_numeric)

# Top Features
  # USING RFE
# control <- rfeControl(functions=rfFuncs, method="cv", number=2)
# results <- rfe(train_df, prices, rfeControl=control)
# importances <- sort(results$fit$importance[,1],decreasing = TRUE)
# print(importances)
# barplot(importances[1:100], cex.names=0.55, cex.axis=0.5)

  # USING BORUTA
boruta.train <- Boruta::Boruta(train_df, prices, doTrace = 2)
print(boruta.train)
important_boruta <- Boruta::getSelectedAttributes(boruta.train,withTentative = FALSE)
final_train <- train_df[important_boruta]
final_test <- test_df[important_boruta]

write.csv(final_train,file="train_boruta.csv",row.names=FALSE)
write.csv(final_test,file="test_boruta.csv",row.names=FALSE)