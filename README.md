# MLearns

Welcome to Machine Learns, we have here implemented various machine learning models using various techniquesand compared them side by side with sklearn's implementations in terms 
of accuracy(for classification) and r2 score(for regression). We use iris dataset for the classification and boston house prices dataset for regression. The output of each file
is one or mode graphs comparing accuracies of our and sklearn's model on various values of hyperparameters. The more close these graphs are to a straight line with a 45 degrees slope, the better is our implementation(considering sklearn's model a benchmark). 

## KNN

<ins>KNNClassifier:</ins>
We have implemented knn using brute force algorithm, majority vote technique for label determination and used minkowski distance, the user can provide his own values for p. The   
model is implemented as a class and we have to pass k and p along with declaration. It has a fit and predict method which takes in numpy arrays or lists. Its accuracies were 
identical to that of sklearn's model

<ins>KNNRegressor:</ins>
We have used same base class for both classification and regression classes, only the predict method is different. We take average of the neighbors y values. Its accuracy is also identical to sklearn's KNN regressor.

## DATA
