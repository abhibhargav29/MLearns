import math
from collections import Counter

import pandas
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score

class KNNClassifier():
    def __init__(self, n_neighbors=5):
        self.K = n_neighbors
    
    def fit(self, data, labels, p=2):
        self.X = data
        self.labels = labels
        self.p=p
    
    def euclideanDistance(self, vec1, vec2):
        p = self.p
        dist = 0
        for i in range(len(vec1)):
            dist += (vec2[i]-vec1[i])**p
        return dist**(1/p)
    
    def nearestPointLabels(self, testVec):
        distances=list()
        for i,vec in enumerate(self.X):
            d = self.euclideanDistance(vec,testVec)
            distances.append((d,i))
        distances.sort(key = lambda x:x[0])
        nearestPoints = distances[:self.K]
        res=[]
        for p in nearestPoints:
            res.append(self.labels[p[1]])
        return res 

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
            
#Compare performace of our classifier vs sklearn's classifier on iris dataset
iris = pandas.read_csv("Data/iris.csv")    
iris=iris.drop("species", axis=1).to_numpy()
iris_L = [0]*50+[1]*50+[2]*50

X_train,X_test,y_train,y_test=model_selection.train_test_split(iris,iris_L,test_size=0.1)

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
