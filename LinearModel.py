import numpy as np
from numpy.random import rand
import pandas
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score, r2_score

#Logistic Regression class
class LogReg:
    #Learning rate and number of iterations will be used in gradient descent
    def __init__(self, C=1.0, learning_rate=0.05, num_iter=100):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.C = C
    
    #The sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    #Fit to data, find weight vector
    def fit(self, X, y):        
        weights = rand(X.shape[1])
        N = len(X)

        # Gradient Descent
        i=0 
        while(i<self.num_iter):
            y_hat = self.sigmoid(np.dot(X, weights)) + ((1/self.C) * sum(weights))
            weights -= self.learning_rate * np.dot(X.T,  y_hat - y) / N            
            i+=1
            
        self.weights = weights
    
    #Predict binary result by using output of sigmoid function
    def predict(self, X):        
        z = np.dot(X, self.weights)
        return [1 if i > 0.5 else 0 for i in (self.sigmoid(z)+(1/self.C)*sum(self.weights))]

if __name__=="__main__":
    #Load iris data and split and convert it to a binary classification problem
    iris = pandas.read_csv("Data/iris.csv")    
    iris=iris.drop("species", axis=1).to_numpy()
    iris_L = [0]*50+[1]*100
    X_train,X_test,y_train,y_test=model_selection.train_test_split(iris,iris_L,test_size=0.1)

    #Test classification model and compare it with sklearn's model
    C_values = [0.01, 0.1, 1, 10, 100]
    accModel1= []
    accModel2= []

    for c in C_values:
        model1 = LogReg(C=c)
        model1.fit(X_train, y_train)
        y_pred = model1.predict(X_test)
        accModel1.append(round(accuracy_score(y_test,y_pred)*100,2))

    for c in C_values:
        model2 = LogisticRegression(C=c, penalty="l1", solver="liblinear", fit_intercept=False)
        model2.fit(X_train, y_train)
        y_pred = model2.predict(X_test)
        accModel2.append(round(accuracy_score(y_test,y_pred)*100,2))

    print(accModel1)
    print(accModel2)
    plt.plot(accModel1, accModel2)
    plt.title("KNN Classifier Accuracy comparison")
    plt.xlabel("accuracy of our knn")
    plt.ylabel("accuracy of sklearn's knn")
    plt.show()
