import pandas
import matplotlib.pyplot as plt
import numpy as np
import math

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score

#Random forest class
class RandomForest():
    def __init__ (self, n_trees=100, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
    
    #Creates the trees
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_classes = len(set(y))
        self.trees = [self.create_tree() for i in range(self.n_trees)]
    
    #Helper for creating single tree from sklearn
    def create_tree(self):
        idxs = np.random.choice(len(self.y), replace=True, size = len(self.y))      
        clf = DecisionTreeClassifier(max_depth=self.max_depth)
        clf.fit(self.X[idxs], self.y[idxs])
        return clf

    #Predict method
    def predict(self, X):
        tree_pred = [tree.predict(X) for tree in self.trees]
        y_pred_float = np.mean(tree_pred, axis=0)
        y_pred=[]
        for y in y_pred_float:
            y_pred.append(round(y))
        return np.array(y_pred)

if __name__=="__main__":
    #Load iris data and split
    iris = pandas.read_csv("Data/iris.csv")    
    iris=iris.drop("species", axis=1).to_numpy()
    iris_L = [0]*50+[1]*50+[2]*50
    X_train,X_test,y_train,y_test=model_selection.train_test_split(iris,iris_L,test_size=0.1)

    #Test classification model and compare it with sklearn's model
    n_values = [10,20,100,150,200]
    accModel1= []
    accModel2= []

    for n in n_values:
        model1 = RandomForest(n_trees=n)
        model1.fit(X_train, np.array(y_train))
        y_pred = model1.predict(X_test)
        accModel1.append(round(accuracy_score(y_test,y_pred)*100,2))

    for n in n_values:
        model2 = RandomForestClassifier(n_estimators=n)
        model2.fit(X_train, y_train)
        y_pred = model2.predict(X_test)
        accModel2.append(round(accuracy_score(y_test,y_pred)*100,2))

    print("Our Accuracies: ", accModel1)
    print("Sklearn's Accuracies: ", accModel2)
    plt.scatter(accModel1, accModel2)
    plt.title("Random Forest Accuracy comparison")
    plt.xlabel("Accuracy of our RF")
    plt.ylabel("Accuracy of sklearn's RF")
    plt.show()
