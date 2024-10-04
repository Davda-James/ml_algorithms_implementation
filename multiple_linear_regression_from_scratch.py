import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split as tts  
class multiple_linear_regression():
    def __init__(self):
        self.intercept=None 
        self.weights=None
    def train(self,X,Y):
        X_train=np.array(X)
        Y_train=np.array(Y)
        extra_rows=np.ones((X.shape[0],1),dtype=float)
        X_train_with_ones=np.hstack((extra_rows,X_train))
        self.weights =np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train_with_ones.T,X_train_with_ones)),X_train_with_ones.T),Y_train)
        self.intercept=self.weights[0]
        self.weights=self.weights[1:]
    def predict(self,X_test):
        X_test_arr=np.array(X_test)
        y_predicted=np.matmul(X_test_arr,self.weights)
        y_predicted=y_predicted+self.intercept
        return y_predicted    
    def squared_error(self,Y_predicted,Y_test):
        return np.sum((Y_predicted-Y_test)**2)
                                 


if __name__=='__main__':
    # Usage 
    mlr=multiple_linear_regression()
    # mlr.train(X_train,Y_train)
    # y_predicted= mlr.predict(X_test)
    # error=mlr.squared_error(y_predicted,Y_test)
    # print(error)
