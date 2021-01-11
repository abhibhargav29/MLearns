import numpy as np
from numpy.random import rand
import pandas
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score, r2_score

#Logistic Regression class
class LogReg:
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


class LinReg() : 
    def __init__(self, learning_rate=0.05, iterations=100): 
        self.learning_rate = learning_rate 
        self.num_iter = iterations 
              
    def fit(self, X, Y):  
        self.X = X 
        self.Y = Y
        self.m, self.n = X.shape

        #Gradient Descent
        self.W = np.zeros(self.n)
        self.b = 0  
        i=0    
        while(i<self.num_iter):   
            Y_pred = X.dot(self.W) + self.b
            self.W = self.W - self.learning_rate * (-(2*(self.X.T).dot(self.Y - Y_pred))/self.m) 
            self.b = self.b - self.learning_rate * (-2*np.sum(self.Y - Y_pred)/self.m)  
            i+=1
      
    def predict(self, X) : 
        return X.dot(self.W) + self.b

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

    print("Our Accuracies:", accModel1)
    print("Sklearn's Accuracies", accModel2)
    plt.scatter(accModel1, accModel2)
    plt.title("Logistic Regression Classifier Accuracy comparison")
    plt.xlabel("accuracy of our log reg")
    plt.ylabel("accuracy of sklearn's log reg")
    plt.show()
    print()

    #Load Boston data and split
    boston = pandas.read_csv("Data/BostonHP.csv").drop("Unnamed: 0", axis=1)
    boston_L = list(boston["label"])
    boston = boston.drop("label",axis=1).to_numpy()
    bos_X_train,bos_X_test,bos_y_train,bos_y_test=model_selection.train_test_split(boston,boston_L,test_size=0.1)

    #Test regression model and compare it with sklearn's model
    scoreModel1 = 0
    scoreModel2 = 0
    
    model1 = LinReg()
    model1.fit(bos_X_train, bos_y_train)
    y_pred = model1.predict(bos_X_test)
    scoreModel1 = round(r2_score(bos_y_test,y_pred), 2)

    model2 = LinearRegression()
    model2.fit(bos_X_train, bos_y_train)
    y_pred = model2.predict(bos_X_test)
    scoreModel2 = round(r2_score(bos_y_test,y_pred), 2)

    print("Our score: ", scoreModel1)
    print("Sklearn's score: ", scoreModel2)
