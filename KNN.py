from collections import Counter
import timeit

import pandas
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score

#Base class for both classification and regression
class KNNBase():
    def __init__(self, n_neighbors=5, p=2):
        self.K = n_neighbors
        self.p = p
    
    def fit(self, data, y):
        self.X = data
        self.Y = y
    
    def minkowskiDistance(self, vec1, vec2):
        dist = 0
        for i in range(len(vec1)):
            dist += abs((vec2[i]-vec1[i])**self.p)
        return dist**(1/self.p)

    def nearestPointLabels(self, testVec):
        distances=list()
        for i,vec in enumerate(self.X):
            d = self.minkowskiDistance(vec,testVec)
            distances.append((d,i))
        distances.sort(key=lambda x:x[0])
        nearestPoints = distances[:self.K]
        res=[]
        for point in nearestPoints:
            res.append(self.Y[point[1]])
        return res


class KNNClassifier(KNNBase):
    def predict(self, x_test):
        y_pred = []
        for vec in x_test:
            res = self.nearestPointLabels(vec)
            res = Counter(res)
            majorLabel = -1
            curr = 0
            for k,v in res.items():
                if(v>curr):
                    majorLabel = k
                    curr=v
                else:
                    pass
            y_pred.append(majorLabel)
        return y_pred


class KNNRegression(KNNBase):
    def predict(self, x_test):
        y_pred = []
        for vec in x_test:
            res = self.nearestPointLabels(vec)
            y_pred.append(sum(res)/self.K)
        return y_pred
    

if __name__=="__main__":
    #Import iris data and split
    iris = pandas.read_csv("Data/iris.csv")    
    iris=iris.drop("species", axis=1).to_numpy()
    iris_L = [0]*50+[1]*50+[2]*50
    X_train,X_test,y_train,y_test=model_selection.train_test_split(iris,iris_L,test_size=0.1)

    #Test classification model and compare it with sklearn's model
    k_values = list(range(1,50,2))
    accModel1= []
    accModel2= []

    for k in k_values:
        model1 = KNNClassifier(n_neighbors=k)
        model1.fit(X_train, y_train)
        y_pred = model1.predict(X_test)
        accModel1.append(accuracy_score(y_test,y_pred)*100)

    for k in k_values:
        model2 = KNeighborsClassifier(n_neighbors=k)
        model2.fit(X_train, y_train)
        y_pred = model2.predict(X_test)
        accModel2.append(accuracy_score(y_test,y_pred)*100)

    plt.plot(accModel1, accModel2)
    plt.title("KNN Classifier Accuracy comparison")
    plt.xlabel("accuracy of our knn")
    plt.ylabel("accuracy of sklearn's knn")
    plt.show()
