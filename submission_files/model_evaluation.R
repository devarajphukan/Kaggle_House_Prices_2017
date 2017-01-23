library("mlbench")
library("e1071")
library("caret")
library("mice")
library("dplyr")
# set.seed(123)
final_train <- read.csv("train_boruta.csv")
final_test <- read.csv("test_boruta.csv")

prices <- final_train$SalePrice
ids <- final_test$Id

final_train <- within(final_train,rm("SalePrice","Id"))
final_test <- within(final_test,rm("Id"))

# Train Test split
split_data <- createDataPartition(y=prices, p = 0.8,list = FALSE)

train_X <- final_train[c(split_data[,1]),]
train_Y <- prices[c(split_data[,1])]

test_X <- final_train[-c(split_data[,1]),]
test_Y <- prices[-c(split_data[,1])]

RMSE_log = function(actual, predicted){
  sqrt(mean((log(actual)-log(predicted))^2))
}

# Model Tuning
# print("Tuning Model")
# doMC::registerDoMC(cores = 4)
# 
# tune_rf <- tune(randomForest::randomForest,
#                 train.x = final_train,
#                 train.y = prices,
#                 ranges=list(ntree=c(300,500,700,1000),
#                             mtry=c(1:15),
#                             nodesize=c(3,5,10,15)))
# print(tune_rf)

# Model Evalutaion
# Random Forest Regression
rf_model <- randomForest::randomForest(train_X, train_Y,ntree=700,mtry=13,nodesize=3)
pred <- as.vector(predict(rf_model,test_X))
print(RMSE_log(test_Y,pred))

# XGBoost Regression
xgb_model <-xgboost::xgboost(data = data.matrix(train_X),
               label = train_Y,
               booster = "gblinear",
               eta = 0.1,
               max_depth = 15,
               nround=2000,
               subsample = 0.4,
               colsample_bytree = 0.3,
               seed = 1,
               eval_metric = "rmse",
               objective = "reg:linear",
               nthread = 4
               )

pred <- predict(xgb_model,data.matrix(test_X))
print(RMSE_log(test_Y,pred))