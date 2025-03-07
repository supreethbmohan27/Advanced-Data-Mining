#Class Tree with 80-80 split without Age and Menopause features

#Install the necessary packages
install.packages('ISLR')
install.packages('rpart')
install.packages('rpart.plot')

#Import the library
library(ISLR)
library(rpart)
library(rpart.plot)
library(caret)
library(pROC)
library(readxl)


# Load data
BreastCancer <- read_excel("C:/Users/supre/Desktop/ADM/Project/ADM Project/breastcancer/breastcancer2.xlsx")

# Link for dataset -> https://archive.ics.uci.edu/dataset/14/breast+cancer


# Perform data pre-processing by removing the null values
BreastCancer <- na.omit(BreastCancer) # Remove NA

# Convert required feature columns to numeric by taking the midpoint of the range
BreastCancer$tumorsize <- sapply(strsplit(BreastCancer$tumorsize, "-"), function(x) mean(as.numeric(x)))
BreastCancer$age <- sapply(strsplit(BreastCancer$age, "-"), function(x) mean(as.numeric(x)))
BreastCancer$invnodes <- sapply(strsplit(BreastCancer$invnodes, "-"), function(x) mean(as.numeric(x)))

# Ensure 'node-caps' is a factor with levels 'yes' and 'no'
BreastCancer$nodecaps <- factor(BreastCancer$nodecaps, levels = c("yes", "no"))

# Ensure 'age' is treated as numeric
BreastCancer$age <- as.numeric(BreastCancer$age)

# Ensure 'inv-nodes' is treated as numeric
BreastCancer$invnodes <- as.numeric(BreastCancer$invnodes)

# 'Class' is the target variable, create a factor
BreastCancer$Class <- factor(BreastCancer$Class)


# Set seed for reproducibility
set.seed(530)
Random.seed <- c("Mersenne-Twister", 530) 



# Split data into training and testing sets with 80:20 split
BreastCancer.split <- sample(1:nrow(BreastCancer), size = nrow(BreastCancer) * 0.8)
BreastCancer.train <- BreastCancer[BreastCancer.split,]
BreastCancer.test <- BreastCancer[-BreastCancer.split,]

# Summary of the dataset
summary(BreastCancer)


#-------------------------------------------------------------------------------
#Model

# Build a classification tree model using 5-fold CV and mini-bucket is 2, which has two ends for each tree
class.cart <- rpart(formula = Class ~ ., data = BreastCancer.train, method = "class", 
                    control = rpart.control(minbucket = 2, xval = 5))

# Visualize the tree
prp(class.cart, roundint = FALSE)

# CP Table
cp.class.param <- class.cart$cptable
cp.class.param

# Initialized arrays to store training, cross-validation, and testing errors for further analysis.
train.err <- double(6)
cv.err <- double(6)
test.err <- double(6)

# Check the length of cp.class.param
n <- nrow(cp.class.param)

# Adjust the lengths of train.err, cv.err, and test.err if necessary, as it throws error if the rows doesn't match
train.err <- train.err[1:n]
cv.err <- cv.err[1:n]
test.err <- test.err[1:n]


# Loop through each row of cp.class.param
for (i in 1:nrow(cp.class.param)) {
  # Extract the CP value from the i-th row of cp.class.param
  alpha <- cp.class.param[i, 'CP']
  
  # Calculate training error
  train.cm <- table(BreastCancer.train$Class, 
                    predict(prune(class.cart, cp=alpha), 
                            newdata = BreastCancer.train, type='class'))
  train.err[i] <- 1 - sum(diag(train.cm)) / sum(train.cm)
  
  # Calculate cross-validation error
  cv.err[i] <- cp.class.param[i, 'xerror'] * cp.class.param[i, 'rel error']
  
  # Calculate test error
  test.cm <- table(BreastCancer.test$Class, 
                   predict(prune(class.cart, cp=alpha), 
                           newdata = BreastCancer.test, type='class'))
  test.err[i] <- 1 - sum(diag(test.cm)) / sum(test.cm)
}
# Comment in detail:
# This loop iterates through each row of the cp.class.param data frame.
# For each row, it extracts the CP value (alpha) and then performs the following tasks:
# 1. Calculate the training error by pruning the decision tree (class.cart) with the given alpha value,
#    predicting on the training data, and computing the error rate.
# 2. Calculate the cross-validation error by multiplying the cross-validation error (xerror) with the
#    relative error (rel error) from the cp.class.param data frame.
# 3. Calculate the test error similarly to the training error but on the testing data.
# The calculated errors are stored in the train.err, cv.err, and test.err arrays for further analysis.



# Print classification error (1 – accuracy) values
train.err
test.err

# Plot training, CV and testing errors at # of Splits/depth
matplot(cp.class.param[,'nsplit'], cbind(train.err, cv.err, test.err), pch=19, col=c("red", "black", "blue"), 
        type="b", ylab="Loss/error", xlab="Depth/# of Splits")
legend("right", c('Train', 'CV', 'Test') ,col=seq_len(3),cex=0.8,fill=c("red", "black", "blue"))

plotcp(class.cart)

# Check CP table, when size of tree =3, the nsplit =4 and CP = 0.0257143 Prune the tree at nsplit =3 defined by the complexity parameter
prune.class.trees <- prune(class.cart, cp=cp.class.param[4,'CP'])
prp(prune.class.trees)

# Unpruned Tree visualization
unprune.class.trees <- unpruned.class.trees <- class.cart
prp(unprune.class.trees)

#-------------------------------------------------------------------------------
#Evaluation Metrics

# For Unpruned Tree

# Calculate confusion table for Unpruned Tree (Test, Train)
conf.mat_unpruned_train <- table(BreastCancer.train$Class, predict(class.cart, type = 'class', newdata = BreastCancer.train))
conf.mat_unpruned_test <- table(BreastCancer.test$Class, predict(class.cart, type = 'class', newdata = BreastCancer.test))

# Unpruned tree on all dataset
unpruned_tree <- rpart(formula = Class ~ ., data = BreastCancer, method = "class", 
                       control = rpart.control(minbucket = 2))

# Calculate confusion table for Unpruned Tree (Full Dataset)
conf.mat_unpruned_all <- table(BreastCancer$Class, predict(unpruned_tree, type = 'class'))

# Print all the confusion matrix from above (Test, Train, Full dataset)
conf.mat_unpruned_train
conf.mat_unpruned_test
conf.mat_unpruned_all

# Metrics for Train dataset: Accuracy, Sensitivity, Specificity, PPV, NPV

# Calculate accuracy for the training set
acc_unpruned_train <- sum(diag(conf.mat_unpruned_train))/sum(conf.mat_unpruned_train)

# Calculate sensitivity for the training set
sensitivity_unpruned_train <- conf.mat_unpruned_train[1, 1] / sum(conf.mat_unpruned_train[1, ])

# Calculate specificity for the training set
specificity_unpruned_train <- conf.mat_unpruned_train[2, 2] / sum(conf.mat_unpruned_train[2, ])

# Calculate positive predictive value for the training set
ppv_unpruned_train <- conf.mat_unpruned_train[1, 1] / sum(conf.mat_unpruned_train[, 1])

# Calculate negative predictive value for the training set
npv_unpruned_train <- conf.mat_unpruned_train[2, 2] / sum(conf.mat_unpruned_train[, 2])



#  Metrics for Test dataset: Accuracy, Sensitivity, Specificity, PPV, NPV

# Calculate accuracy for the test set
acc_unpruned_test <- sum(diag(conf.mat_unpruned_test))/sum(conf.mat_unpruned_test)

# Calculate sensitivity for the test set
sensitivity_unpruned_test <- conf.mat_unpruned_test[1, 1] / sum(conf.mat_unpruned_test[1, ])

# Calculate specificity for the test set
specificity_unpruned_test <- conf.mat_unpruned_test[2, 2] / sum(conf.mat_unpruned_test[2, ])

# Calculate positive predictive value for the test set
ppv_unpruned_test <- conf.mat_unpruned_test[1, 1] / sum(conf.mat_unpruned_test[, 1])

# Calculate negative predictive value for the test set
npv_unpruned_test <- conf.mat_unpruned_test[2, 2] / sum(conf.mat_unpruned_test[, 2])



#  Metrics for Full dataset: Accuracy, Sensitivity, Specificity, PPV, NPV

# Accuracy for unpruned tree on all dataset
acc_unpruned_all <- sum(diag(conf.mat_unpruned_all))/sum(conf.mat_unpruned_all)

# Sensitivity for unpruned tree on all dataset
sensitivity_unpruned_all <- conf.mat_unpruned_all[1, 1] / sum(conf.mat_unpruned_all[1, ])

# Specificity for unpruned tree on all dataset
specificity_unpruned_all <- conf.mat_unpruned_all[2, 2] / sum(conf.mat_unpruned_all[2, ])

# Positive Predictive Value (PPV) for unpruned tree on all dataset
ppv_unpruned_all <- conf.mat_unpruned_all[1, 1] / sum(conf.mat_unpruned_all[, 1])

# Negative Predictive Value (NPV) for unpruned tree on all dataset
npv_unpruned_all <- conf.mat_unpruned_all[2, 2] / sum(conf.mat_unpruned_all[, 2])


# For Pruned Tree

#confusion matrix for Train, Test and Full dataset
conf.mat_pruned_train <- table(BreastCancer.train$Class, predict(prune.class.trees, type = 'class', newdata = BreastCancer.train))
conf.mat_pruned_test <- table(BreastCancer.test$Class, predict(prune.class.trees, type = 'class', newdata = BreastCancer.test))
conf.mat_pruned_all <- table(BreastCancer$Class, predict(prune.class.trees, type = 'class', newdata = BreastCancer))

#print all the above confusion matrix
conf.mat_pruned_train
conf.mat_pruned_test
conf.mat_pruned_all


#  Metrics for Train dataset: Accuracy, Sensitivity, Specificity, PPV, NPV

# Calculate accuracy for the training set
acc_pruned_train <- sum(diag(conf.mat_pruned_train))/sum(conf.mat_pruned_train)

# Calculate sensitivity for the training set
sensitivity_pruned_train <- conf.mat_pruned_train[1, 1] / sum(conf.mat_pruned_train[1, ])

# Calculate specificity for the training set
specificity_pruned_train <- conf.mat_pruned_train[2, 2] / sum(conf.mat_pruned_train[2, ])

# Calculate positive predictive value for the training set
ppv_pruned_train <- conf.mat_pruned_train[1, 1] / sum(conf.mat_pruned_train[, 1])

# Calculate negative predictive value for the training set
npv_pruned_train <- conf.mat_pruned_train[2, 2] / sum(conf.mat_pruned_train[, 2])


#  Metrics for Test dataset: Accuracy, Sensitivity, Specificity, PPV, NPV

# Calculate accuracy for the test set
acc_pruned_test <- sum(diag(conf.mat_pruned_test))/sum(conf.mat_pruned_test)

# Calculate sensitivity for the test set
sensitivity_pruned_test <- conf.mat_pruned_test[1, 1] / sum(conf.mat_pruned_test[1, ])

# Calculate specificity for the test set
specificity_pruned_test <- conf.mat_pruned_test[2, 2] / sum(conf.mat_pruned_test[2, ])

# Calculate positive predictive value for the test set
ppv_pruned_test <- conf.mat_pruned_test[1, 1] / sum(conf.mat_pruned_test[, 1])

# Calculate negative predictive value for the test set
npv_pruned_test <- conf.mat_pruned_test[2, 2] / sum(conf.mat_pruned_test[, 2])


#  Metrics for Full dataset: Accuracy, Sensitivity, Specificity, PPV, NPV

# Calculate accuracy for the entire dataset
acc_pruned_all <- sum(diag(conf.mat_pruned_all))/sum(conf.mat_pruned_all)

# Calculate sensitivity for the entire dataset
sensitivity_pruned_all <- conf.mat_pruned_all[1, 1] / sum(conf.mat_pruned_all[1, ])

# Calculate specificity for the entire dataset
specificity_pruned_all <- conf.mat_pruned_all[2, 2] / sum(conf.mat_pruned_all[2, ])

# Calculate positive predictive value for the entire dataset
ppv_pruned_all <- conf.mat_pruned_all[1, 1] / sum(conf.mat_pruned_all[, 1])

# Calculate negative predictive value for the entire dataset
npv_pruned_all <- conf.mat_pruned_all[2, 2] / sum(conf.mat_pruned_all[, 2])


# Print the metrics for unpruned and pruned trees for both train, test and full datasets

# Unpruned Tree
# Print results for unpruned train on all dataset
acc_unpruned_train
sensitivity_unpruned_train
specificity_unpruned_train
ppv_unpruned_train
npv_unpruned_train

# Print results for unpruned tree on test dataset
acc_unpruned_test
sensitivity_unpruned_test
specificity_unpruned_test
ppv_unpruned_test
npv_unpruned_test

# Print results for unpruned tree on all dataset
acc_unpruned_all
sensitivity_unpruned_all
specificity_unpruned_all
ppv_unpruned_all
npv_unpruned_all

# Pruned Tree
# Print results for Pruned tree on train dataset
acc_pruned_train
sensitivity_pruned_train
specificity_pruned_train
ppv_pruned_train
npv_pruned_train

# Print results for Pruned tree on test dataset
acc_pruned_test
sensitivity_pruned_test
specificity_pruned_test
ppv_pruned_test
npv_pruned_test

# Print results for Pruned tree on full dataset
acc_pruned_all
sensitivity_pruned_all
specificity_pruned_all
ppv_pruned_all
npv_pruned_all


#Plot ROC-AUc Curve for Unpruned and Pruned Tree

#Initialize a new plot
plot.new()

# Unpruned Classification Tree

# Plot ROC curve for unpruned classification tree


# Calculate AUC for train dataset separately for classification tree
train.prob_tree <- predict(class.cart, newdata = BreastCancer.train, type = "prob")[,2] # Probability of class 'Yes'
train.roc_tree = roc(BreastCancer.train$Class, train.prob_tree)
plot(train.roc_tree, col="brown", add = TRUE, position = c(0.8, 0.1))
# Add a smoothed ROC for classification tree
plot.roc(smooth(train.roc_tree), col = "red", add = TRUE)


# Calculate AUC for test dataset seperately for classification tree
test.prob_tree_unpruned <- predict(class.cart, newdata = BreastCancer.test, type = "prob")[,2]
test.roc_tree_unpruned <- roc(BreastCancer.test$Class, test.prob_tree_unpruned, legacy.axes = TRUE)
plot(test.roc_tree_unpruned, col = "black", position = c(0.8, 0.1))
# Add smoothed ROC curve for unpruned classification tree
plot.roc(smooth(test.roc_tree_unpruned), col = "blue", add = TRUE)


# Calculate AUC for Full dataset seperately for classification tree
all.prob_tree <- predict(class.cart, newdata = BreastCancer, type = "prob")[,2] # Probability of class 'Yes'
all.roc_tree = roc(BreastCancer$Class, all.prob_tree)
plot(all.roc_tree, col="grey", add = TRUE, position = c(0.8, 0.1))
# Add a smoothed ROC for classification tree
plot.roc(smooth(all.roc_tree), col = "green", add = TRUE)

# Adjust labels
legend("bottomright", legend = c(
  paste("Unpruned Train (Empirical)", "AUC =", round(train.roc_tree$auc, 2)),
  paste("Unpruned Train (Smoothed)"),
  paste("Unpruned Test (Empirical)", "AUC =", round(test.roc_tree_unpruned$auc, 2)),
  paste("Unpruned Test (Smoothed)"),
  paste("Unpruned All (Empirical)", "AUC =", round(all.roc_tree$auc, 2)),
  paste("Unpruned All (Smoothed)")
), col = c("black", "blue", "brown", "red", "grey", "green"), lwd = 2, cex = 0.6)



# Pruned Tree
# Plot ROC curve for Pruned classification tree

# Calculate AUC for pruned classification tree train set
train.prob_tree_pruned <- predict(prune.class.trees, newdata = BreastCancer.train, type = "prob")[,2]
train.roc_tree_pruned <- roc(BreastCancer.train$Class, train.prob_tree_pruned, drop_intermediate = FALSE)
plot(train.roc_tree_pruned, col = "brown", add = TRUE, position = c(0.8, 0.1))
# Add a smoothed ROC for classification tree
plot.roc(smooth(train.roc_tree_pruned), col = "red", add = TRUE)

# Calculate AUC for pruned classification tree test set
test.prob_tree_pruned <- predict(prune.class.trees, newdata = BreastCancer.test, type = "prob")[,2]
test.roc_tree_pruned <- roc(BreastCancer.test$Class, test.prob_tree_pruned, legacy.axes = TRUE)
plot(test.roc_tree_pruned, col = "black", position = c(0.8, 0.1))
# Add a smoothed ROC for classification tree
plot.roc(smooth(test.roc_tree_pruned), col = "blue", add = TRUE)


# Calculate AUC for pruned classification tree full dataset
all.prob_tree_pruned <- predict(prune.class.trees, newdata = BreastCancer, type = "prob")[,2]
all.roc_tree_pruned <- roc(BreastCancer$Class, all.prob_tree_pruned)
plot(all.roc_tree_pruned, col = "grey", add = TRUE, position = c(0.8, 0.1))
plot.roc(smooth(all.roc_tree_pruned), col = "green", add = TRUE)


# Adjust labels
legend("bottomright", legend = c(
  paste("Pruned Train (Empirical)", "AUC =", round(train.roc_tree_pruned$auc, 2)),
  paste("Pruned Train (Smoothed)"),
  paste("Pruned Test (Empirical)", "AUC =", round(test.roc_tree_pruned$auc, 2)),
  paste("Pruned Test (Smoothed)"),
  paste("Pruned All (Empirical)", "AUC =", round(all.roc_tree_pruned$auc, 2)),
  paste("Pruned All (Smoothed)")
), col = c("black", "blue", "brown", "red", "grey", "green"), lwd = 2, cex = 0.8)
