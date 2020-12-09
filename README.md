# MLearns
Welcome to Machine Learns, we have here implemented various machine learning models using various techniques and compared them side by side with sklearn's implementations in terms 
of accuracy(for classification) and r2 score(for regression). We use iris dataset for the classification and boston house prices dataset for regression. The output of each file
is one or mode graphs comparing accuracies of our and sklearn's model on various values of hyperparameters. The more close these graphs are to the straight line x=y, the better is 
our implementation(considering sklearn's model a benchmark). 

## KNN
<ins>KNNClassifier:</ins>
We have implemented knn using brute force algorithm, majority vote technique for label determination and used minkowski distance, the user can provide his own values for p. The
model is implemented as a class and we have to pass k and p along with declaration. It has a fit and predict method which takes in numpy arrays or lists. Its accuracies were 
identical to that of sklearn's model

<ins>KNNRegressor:</ins>
We have used same base class for both classification and regression classes, only the predict method is different. We take average of the neighbors y values. Its accuracy is also identical to sklearn's KNN regressor.

## Linear Model
<ins>Logistic Regression:</ins>
We have implemented our own logistic regression model in LogReg class with L1 regularization. The accuracies for different values of hyperparameters were almost similar to 
sklearn's model on setting the parameters similarly like regularization, solver algorithm, etc. In fact, they were identical for most values of hyperparameters.

## Naive Bayes
We have implemented gaussian naive bayes, in this algorithm we use bayes theorem and calculate probabilities assuming that features are gaussian distributed. We have not used 
variable smoothing like sklearn and thus compared our accuracy with that of sklearn's GaussianNB at var_smoothing=0. Naive Bayes is rarely used for regression so we have not 
implemented a regression class.

## Data
<ins>Iris:</ins>
The dataset is taken for iris flower dataset kaggle, it is a very famous toy dataset. It needs not to be normalized because all parameters are in inches only. We have 3 class 
classification with balanced data for each class. Link: https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv

<ins>BostonHP:</ins>
This is the boston house prices dataset which has 12-features or variables for 506 houses and we have to predict the house price. It is already stored in standardized form and 
just needs to be loaded, split and given to the model. It comes with sklearn in sklearn.datasets, we have loaded it, normalized it and stored it in the csv file which we load in 
our models files.
