import math
from collections import Counter
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score

class KNNClassifier():
    def __init__(self, n_neighbors=5):
        self.K = n_neighbors
    
    def fit(self, data, labels):
        self.X = data
        self.labels = labels
    
    def euclideanDistance(self, vec1, vec2):
        dist = 0
        for i in range(len(vec1)):
            dist += (vec2[i]-vec1[i])**2
        return math.sqrt(dist)
    
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
            
iris = pandas.read_csv("iris.csv")
iris=iris.drop("species", axis=1).to_numpy()
iris_L = [0]*50+[1]*50+[2]*50

X_train,X_test,y_train,y_test=model_selection.train_test_split(iris,iris_L,test_size=0.1)

model1 = KNNClassifier(n_neighbors=5)
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(y_pred)
print(y_test)
print()

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
y_pred = list(model2.predict(X_test))
print(accuracy_score(y_test,y_pred))
print(y_pred)
print(y_test)
