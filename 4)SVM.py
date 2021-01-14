import pandas
import matplotlib.pyplot as plt
import numpy as np

import cvxopt
import cvxopt.solvers as solvers

from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import accuracy_score

class KernelSVM():
    def __init__(self, C=1.0, kernel="lin"):
        if kernel in ["lin","poly","rbf"]:
            self.kernel = self.assign_Kernel(kernel)
        else:
            self.kernel = self.linear_kernel 
        self.C = C

    def assign_Kernel(self, kernel):
        if(kernel=="lin"):
            return self.linear_kernel
        elif(kernel=="poly"):
            return self.polynomial_kernel
        else:
            return self.rbf_kernel

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel(self, x1, x2, p=3):
        return (1 + np.dot(x1, x2)) ** p

    def rbf_kernel(self, x1, x2, gamma=1.0):
        distance = np.linalg.norm(x1-x2)**2
        return np.exp(-gamma*distance)

    def fit(self, X, y, kernel_parameter=None):
        n_samples, n_features = X.shape

        #Kernel Matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if(kernel_parameter==None):
                    K[i][j] = self.kernel(X[i],X[j])
                else:
                    K[i][j] = self.kernel(X[i],X[j], kernel_parameter)

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc="d")
        b = cvxopt.matrix(0.0)
    
        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * self.C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        
        #Find Solution
        solution = solvers.qp(P,q,G,h,A=A,b=b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[ind]
        self.sv = X[ind]
        self.sv_y = y[ind]

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv]) 
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == self.linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))


if __name__=="__main__":
    #Load iris data and split and convert it to a binary classification problem
    iris = pandas.read_csv("Data/iris.csv")    
    iris=iris.drop("species", axis=1).to_numpy()[:100]
    iris_L = [0]*50+[1]*50
    X_train,X_test,y_train,y_test=model_selection.train_test_split(iris,iris_L,test_size=0.1)

    #Test classification model and compare it with sklearn's model
    C_values = [0.01, 0.1, 1, 10, 100]
    accModel1= []
    accModel2= []

    for c in C_values:
        model1 = KernelSVM(C=c, kernel="poly")
        model1.fit(X_train, np.array(y_train))
        y_pred = model1.predict(X_test)
        accModel1.append(round(accuracy_score(y_test,y_pred)*100,2))

    for c in C_values:
        model2 = SVC(C=c, kernel="poly", degree=3, gamma=1.0)
        model2.fit(X_train, y_train)
        y_pred = model2.predict(X_test)
        accModel2.append(round(accuracy_score(y_test,y_pred)*100,2))

    print("Our Accuracies:", accModel1)
    print("Sklearn's Accuracies", accModel2)
    
    
