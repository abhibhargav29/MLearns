# MLearns
<p align="justify">
Welcome to Machine Learns, we have here implemented various machine learning models using various techniques and compared them side by side with sklearn's implementations in 
terms of accuracy(for classification) and r2 score(for regression). We use iris dataset for the classification and boston house prices dataset for regression. The output of each 
file is accuracies of our and sklearn's model and one or more scatter plots comparing accuracies of our and sklearn's model as on various values of hyperparameters. The more 
close these points are to the straight line y=x, the better is our implementation(considering sklearn's model a benchmark). 
</p>

## KNN
<p align="justify">
<ins>KNNClassifier:</ins>
We have implemented knn using brute force algorithm, majority vote technique for label determination and used minkowski distance(the user can provide his own values for p). The
model is implemented as a class and we have to pass k and p along with declaration. It has a fit and predict method which takes in numpy arrays or lists. Its accuracies were 
identical to that of sklearn's model. We do not get to see them properly in the scatter plot as both models gave almost 100% accuracies for all values of k and thus we only see 
one point(which infact is a superposition of many points).
</p>

<p align="justify">
<ins>KNNRegressor:</ins>
We have used same base class for both classification and regression classes, only the predict method is different. In Classification the predict method does majority vote while 
in regression we do a simple average. Other methods are same due to polymorphism because we do not make use of the discrete/continuous nature of the class in any of other 
methods. Its r2 score is identical to sklearn's KNN regressor and we can see in the scatter plot that the points are in a straight line y=x.
</p>

## Linear Model
<p align="justify">
<ins>Logistic Regression:</ins>
We have implemented our own logistic regression model in LogReg class with L1 regularization. Our logistic regression class can only do binary classification as we have not used 
1 vs rest technique. The accuracies for different values of hyperparameters were almost similar to sklearn's model on setting the parameters similarly like regularization, 
solver algorithm, etc. In fact, they were identical for most values of hyperparameters, the same is not too clear from the scatter plot.
</p>

<p align="justify">
<ins>Linear Regression:</ins>
The class LinReg implements our linear regression model. We have used gradient descent for linear regression as well to find the best fit line. The r2 score of our model is 
identical to that of sklearn's. Here we find two things, the weights and the constant(bias) using gradient descent unlike we did in logistic regression where we just find out 
the weights.
</p>

## Naive Bayes
<p align="justify">
We have implemented gaussian naive bayes, in this algorithm we use bayes theorem and calculate probabilities assuming that features are gaussian distributed. We have not used 
variable smoothing like sklearn and thus compared our accuracy with that of sklearn's GaussianNB at var_smoothing=0 and their is no graph for different smoothing. Naive Bayes is 
rarely used for regression so we have not implemented a regression class.
</p>

## SVM
<p align="justify">
We have implemented kernel SVM with linear, polynomial and rbf kernel. This SVM class can only do binary classification. We have used cvxopt library to find solutions to the 
equation. cvxopt is a very good library to find the solution to convex optimization problems. The fit method forms the kernel matrix and computes the solution for the weight 
vector and intercept. The project method finds the projection of point on the plane and using the sign of that projection we find the class of that point. The regularization 
term is given while creating an object of the model class and the additional hyperparameter depending on kernel is given during fit. The accuracies of this model were not 
comparable to sklearn's but the hyperparameter-accuracy comparison was similar.
</p>

## Decision Tree
We have implemeted decision tree using cart algorithm and gini impurity as the metric for split.

## Random Forest

## Data
<p align="justify">
<ins>Iris:</ins>
The dataset is taken for iris flower dataset kaggle, it is a very famous toy dataset. It needs not to be normalized because all parameters are in inches only. We have 3 class 
classification with balanced data for each class and 4 features, there are a total of 150 rows, 50 for each class.
</p>

<p align="justify">
<ins>BostonHP:</ins>
This is the boston house prices dataset which has 12 features for 506 houses and we have to predict the house price. It is already stored in standardized form and just needs to 
be loaded, split and given to the model. It comes with sklearn in sklearn.datasets, we have loaded it, normalized it and stored it as the csv file which we load in our models 
files.
</p>
