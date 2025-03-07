#XGBoost with 50-50 split without Age and Menopause features

#install the necessary packages
install.packages('ISLR')
install.packages('rpart')
install.packages('rpart.plot')
install.packages('xgboost')
library(ISLR)
library(rpart)
library(rpart.plot)
library(caret)
library(pROC)
library(readxl)
library(xgboost)

# Load data
BreastCancer <- read_excel("C:/Users/supre/Desktop/ADM/Project/ADM Project/breastcancer/breastcancer2.xlsx")

#link for dataset -> https://archive.ics.uci.edu/dataset/14/breast+cancer


#perform data pre-processing by removing the null values
BreastCancer <- na.omit(BreastCancer) # Remove NA


#Apply mean function to take mean values which cell or value is empty or null
BreastCancer$tumorsize <- sapply(strsplit(BreastCancer$tumorsize, "-"), function(x) mean(as.numeric(x)))
BreastCancer$invnodes <- sapply(strsplit(BreastCancer$invnodes, "-"), function(x) mean(as.numeric(x)))

# Factorize non-numeric columns
BreastCancer$nodecaps <- factor(BreastCancer$nodecaps, levels = c("yes", "no"))
BreastCancer$breast <- factor(BreastCancer$breast, levels = c("left", "right"))
BreastCancer$breastquad <- factor(BreastCancer$breastquad, levels = c("leftup", "leftlow", "rightup", "rightlow", "central"))
BreastCancer$irradiat <- factor(BreastCancer$irradiat, levels = c("yes", "no"))

# Convert factor levels to numeric values
BreastCancer$tumorsize <- as.numeric(BreastCancer$tumorsize)
BreastCancer$invnodes <- as.numeric(BreastCancer$invnodes)
BreastCancer$nodecaps <- as.numeric(BreastCancer$nodecaps)
BreastCancer$breast <- as.numeric(BreastCancer$breast)
BreastCancer$breastquad <- as.numeric(BreastCancer$breastquad)
BreastCancer$irradiat <- as.numeric(BreastCancer$irradiat)

# Convert "no-recurrence-events" to 0 and "recurrence-events" to 1
BreastCancer$Class <- as.numeric(ifelse(BreastCancer$Class == "recurrence-events", 1, 0))

# Set seed for reproducibility
set.seed(530)
Random.seed <- c("Mersenne-Twister", 530) 

# Remove 'age' and 'menopause' columns
BreastCancer <- BreastCancer[, !names(BreastCancer) %in% c("age", "menopause")]


BreastCancer <- na.omit(BreastCancer)


# Split data into training and testing sets with 50:50 split
BreastCancer.split <- sample(1:nrow(BreastCancer), size = nrow(BreastCancer) * 0.5)
BreastCancer.train <- BreastCancer[BreastCancer.split,]
BreastCancer.test <- BreastCancer[-BreastCancer.split,]

# Convert data to matrix
dtrain <- xgb.DMatrix(data = as.matrix(BreastCancer.train[, -1]), label = BreastCancer.train$Class)
dtest <- xgb.DMatrix(data = as.matrix(BreastCancer.test[, -1]))

# Set hyperparameters
params <- list(
  objective = "binary:logistic",  # For binary classification
  eval_metric = "error"            # Evaluation metric
)

# Define the XGBoost cross-validation model
xgb_model_cv <- xgb.cv(
  data = dtrain,                         # Specify the training data
  params = params,                       # Specify the model parameters
  nfold = 5,                             # Number of folds for cross-validation
  nrounds = 6,                           # Number of boosting rounds
  early_stopping_rounds = 5,             # Early stopping rounds to prevent overfitting
  seed = 530,                            # Set seed for reproducibility
  metrics = "error",                     # Evaluation metric for cross-validation
  maximize = FALSE                       # Minimize the evaluation metric
)


# Get the cross-validation error for each boosting round
cv_errors <- xgb_model_cv$evaluation_log$test_error_mean

# Find the optimal number of boosting rounds that minimizes the cross-validation error
best_nrounds <- which.min(cv_errors)

best_nrounds

# Train the final model with the optimal number of boosting rounds
xgb_model <- xgboost(
  data = dtrain,       # Specify the training data
  params = params,     # Specify the model parameters
  nrounds = best_nrounds  # Use the optimal number of boosting rounds determined during tuning
)


# Make predictions on the test data
predictions <- predict(xgb_model, dtest)

# Convert predictions to class labels
predicted_labels <- ifelse(predictions > 0.5, 1, 0)

# Evaluate the model
accuracy <- mean(predicted_labels == BreastCancer.test$Class)


# Predictions on the train set
predictions_train <- predict(xgb_model, as.matrix(BreastCancer.train[, -1]))
predictions_train <- ifelse(predictions_train > 0.5, 1, 0)

# Predictions on the test set
predictions_test <- predict(xgb_model, as.matrix(BreastCancer.test[, -1]))
predictions_test <- ifelse(predictions_test > 0.5, 1, 0)

# Predictions on Full Dataset
predictions_all <- predict(xgb_model, as.matrix(BreastCancer[, -1]))
predictions_all <- ifelse(predictions_all > 0.5, 1, 0)


#Confusion Matrix
conf_mat_train <- table(BreastCancer.train$Class, predictions_train)
conf_mat_test <- table(BreastCancer.test$Class, predictions_test)
conf_mat_all <- table(BreastCancer$Class, predictions_all)


# Confusion matrix for train set
acc_xgb_train <- sum(diag(conf_mat_train)) / sum(conf_mat_train)
sensitivity_xgb_train <- conf_mat_train[1, 1] / sum(conf_mat_train[1, ])
specificity_xgb_train <- conf_mat_train[2, 2] / sum(conf_mat_train[2, ])
ppv_xgb_train <- conf_mat_train[1, 1] / sum(conf_mat_train[, 1])
npv_xgb_train <- conf_mat_train[2, 2] / sum(conf_mat_train[, 2])


# Confusion matrix for test set
acc_xgb_test <- sum(diag(conf_mat_test)) / sum(conf_mat_test)
sensitivity_xgb_test <- conf_mat_test[1, 1] / sum(conf_mat_test[1, ])
specificity_xgb_test <- conf_mat_test[2, 2] / sum(conf_mat_test[2, ])
ppv_xgb_test <- conf_mat_test[1, 1] / sum(conf_mat_test[, 1])
npv_xgb_test <- conf_mat_test[2, 2] / sum(conf_mat_test[, 2])

# Confusion matrix for all dataset
acc_xgb_all <- sum(diag(conf_mat_all)) / sum(conf_mat_all)
sensitivity_xgb_all <- conf_mat_all[1, 1] / sum(conf_mat_all[1, ])
specificity_xgb_all <- conf_mat_all[2, 2] / sum(conf_mat_all[2, ])
ppv_xgb_all <- conf_mat_all[1, 1] / sum(conf_mat_all[, 1])
npv_xgb_all <- conf_mat_all[2, 2] / sum(conf_mat_all[, 2])


#print confusion matrix
conf_mat_train
conf_mat_test
conf_mat_all

# Print the metrics for the XGBoost model
acc_xgb_train
sensitivity_xgb_train
specificity_xgb_train
ppv_xgb_train
npv_xgb_train

acc_xgb_test
sensitivity_xgb_test
specificity_xgb_test
ppv_xgb_test
npv_xgb_test

acc_xgb_all
sensitivity_xgb_all
specificity_xgb_all
ppv_xgb_all
npv_xgb_all

# Predict probabilities for train, test, and all datasets
train_pred_prob <- predict(xgb_model, as.matrix(BreastCancer.train[, -1]))
test_pred_prob <- predict(xgb_model, as.matrix(BreastCancer.test[, -1]))
all_pred_prob <- predict(xgb_model, as.matrix(BreastCancer[, -1]))

# Calculate ROC curves
train_roc <- roc(BreastCancer.train$Class, train_pred_prob)
test_roc <- roc(BreastCancer.test$Class, test_pred_prob)
all_roc <- roc(BreastCancer$Class, all_pred_prob)

# Check the number of points in the ROC curve
num_points <- length(train_roc$thresholds)
print(num_points)

summary(train_pred_prob)

# Check the range
min_prob <- min(train_pred_prob)
max_prob <- max(train_pred_prob)

print(min_prob)
print(max_prob)

# Smooth ROC curves
smoothed_train_roc <- smooth(train_roc)
smoothed_test_roc <- smooth(test_roc)
smoothed_all_roc <- smooth(all_roc)

# Plot ROC curves
plot(train_roc, col = "blue", main = "ROC Curves for XGBoost",
     xlab = "False Positive Rate", ylab = "True Positive Rate")
plot(smoothed_train_roc, col = "lightblue", add = TRUE)
plot(test_roc, col = "red", add = TRUE)
plot(smoothed_test_roc, col = "pink", add = TRUE)
plot(all_roc, col = "green", add = TRUE)
plot(smoothed_all_roc, col = "lightgreen", add = TRUE)

# Calculate AUC for each curve
train_auc <- round(auc(train_roc), 2)
test_auc <- round(auc(test_roc), 2)
all_auc <- round(auc(all_roc), 2)
smoothed_train_auc <- round(auc(smoothed_train_roc), 2)
smoothed_test_auc <- round(auc(smoothed_test_roc), 2)
smoothed_all_auc <- round(auc(smoothed_all_roc), 2)

# Add legend with AUC values
legend("bottomright", legend = c(
  paste("Train (Empirical)", "AUC =", train_auc),
  paste("Train (Smoothed)", "AUC =", smoothed_train_auc),
  paste("Test (Empirical)", "AUC =", test_auc),
  paste("Test (Smoothed)", "AUC =", smoothed_test_auc),
  paste("All (Empirical)", "AUC =", all_auc),
  paste("All (Smoothed)", "AUC =", smoothed_all_auc)
), col = c("blue", "lightblue", "red", "pink", "green", "lightgreen"), lwd = 2, cex = 0.8)




