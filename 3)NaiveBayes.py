import pandas
import matplotlib.pyplot as plt
import math
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import accuracy_score

class NaiveBayes():
    def __init__(self):
        pass
    
    #Separate different classes
    def separateByClass(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        self.classes = classes
        self.class_counts = dict(zip(classes, counts))
        self.class_freq = {}
        datasets = {}

        for c in self.classes:
            datasets[c] = X[np.argwhere(y==c), :]
            self.class_freq[c] = self.class_counts[c]/sum(list(self.class_counts.values()))
        return datasets

    #Calculate mean and standard deviation for each class
    def fit(self, X, y):
        datasets = self.separateByClass(X, y)
        self.means = {}
        self.std = {}
        for c in self.classes:
            self.means[c] = np.mean(datasets[c], axis=0)[0]
            self.std[c] = np.std(datasets[c], axis=0)[0]
    
    #Using pdf of a Gaussian distribution
    def calculateProb(self, x, mean, std):
        exponent = math.exp(-((x-mean)**2)/(2*(std**2)))
        return (1/math.sqrt(2*math.pi)*std)*exponent

    #Calculation of probability of belonging to each class
    def predictProb(self, X):
        classProb = {c:math.log(self.class_freq[c], math.e) for c in self.classes}
        for c in self.classes:
            temp = classProb[c]
            for i in range(len(X)):
                temp += math.log(self.calculateProb(X[i], self.means[c][i], self.std[c][i]), math.e)
            classProb[c] = math.exp(temp)
        return classProb

    #Final prediction
    def predict(self, X):
        y_pred = []
        for x in X:
            pred_class = None
            max_prob = 0
            for c, prob in self.predictProb(x).items():
                if prob>max_prob:
                    max_prob = prob
                    pred_class = c
            y_pred.append(pred_class)
        return y_pred

if __name__=="__main__":
    #Load iris data and split
    iris = pandas.read_csv("Data/iris.csv")    
    iris=iris.drop("species", axis=1).to_numpy()
    iris_L = [0]*50+[1]*50+[2]*50
    X_train,X_test,y_train,y_test=model_selection.train_test_split(iris,iris_L,test_size=0.1)

    #Predict on iris data
    model1 = NaiveBayes()
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    accModel1 = round(accuracy_score(y_test, y_pred)*100,2)

    model2 = GaussianNB(var_smoothing=0)
    model2.fit(X_train, y_train)
    y_pred = model2.predict(X_test)
    accModel2 = round(accuracy_score(y_test, y_pred)*100,2)

    print("Our Accuracy: ",accModel1)
    print("Sklearn's Accuracy", accModel2)
