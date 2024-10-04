import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score
import random 
class minibatchRegressor:
    def __init__(self,batch_size,learning_rate=0.01,epochs=100):
        self.intercept=None 
        self.weights=None 
        self.epochs=epochs
        self.eta=learning_rate
        self.batch_size=batch_size
    def train(self,X_train,Y_train):
        self.weights=np.ones(X_train.shape[1])
        self.intercept=0
        for i in range(self.epochs):
            for j in range(X_train.shape[0]//self.batch_size):
                random_indexes=random.sample(range(X_train.shape[0]),self.batch_size)
                X_subset=X_train[random_indexes]
                y_hat=np.dot(X_subset,self.weights)+self.intercept
                Y_sub_train_hat=Y_train[random_indexes]-y_hat
                intercept_slope=-2*np.mean(Y_sub_train_hat)
                weights_slope=-2*np.dot(Y_sub_train_hat,X_subset)
                self.intercept=self.intercept-self.eta*intercept_slope 
                self.weights=self.weights-self.eta*weights_slope
    def predict(self,X_test):
        return np.dot(X_test,self.weights)+self.intercept
    def calculate_accuracy_score(self,Y_test,Y_predict):
        return r2_score(Y_test,Y_predict)



if __name__=='__main__':
    X,Y=load_diabetes(return_X_y=True)
    X_train,X_test,Y_train,Y_test=tts(X,Y,random_state=2,test_size=0.2)
    minibatch=minibatchRegressor(10)
    minibatch.train(X_train,Y_train)
    Y_predicted=minibatch.predict(X_test)
    score=minibatch.calculate_accuracy_score(Y_test,Y_predicted)
    print(score)
