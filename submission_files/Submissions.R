library("mlbench")
library("caret")
library("mice")
library("dplyr")

final_train <- read.csv("train_boruta.csv")
final_test <- read.csv("test_boruta.csv")

prices <- final_train$SalePrice
ids <- final_test$Id

final_train <- within(final_train,rm("SalePrice","Id"))
final_test <- within(final_test,rm("Id"))

# # Final Prediction
# xgb_model_final <-xgboost::xgboost(data = data.matrix(final_train),
#                              label = prices,
#                              booster = "gblinear",
#                              eta = 0.1,
#                              max_depth = 10,
#                              nround=200,
#                              subsample = 0.4,
#                              colsample_bytree = 0.3,
#                              seed = 1,
#                              eval_metric = "rmse",
#                              objective = "reg:linear",
#                              nthread = 4,
#                              verbose = FALSE
# )
# pred <- predict(xgb_model_final,data.matrix(final_test))

rf_model <- randomForest::randomForest(final_train, prices,ntree=700,mtry=13,nodesize=3)
pred <- as.vector(predict(rf_model,final_test))

result_df <- data.frame(Id=ids,SalePrice=pred)
write.csv(result_df,file="results.csv",row.names=FALSE)
