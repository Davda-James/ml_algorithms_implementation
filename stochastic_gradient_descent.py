import numpy as np 
from sklearn.metrics import r2_score 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split as tts 
class SGDregressor:
    def __init__(self,epochs=100,learning_rate=0.01):
        self.eta=learning_rate
        self.epochs=epochs
        self.intercept=None 
        self.weights=None
    def train(self,X_train,Y_train):
        self.intercept=1
        self.weights=np.ones((X_train.shape[1],))
        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                random_index=np.random.randint(0,X_train.shape[0])
                y_hat=np.dot(X_train[random_index],self.weights)+self.intercept
                y_sub_train_hat=Y_train[random_index]-y_hat
                intercept_slope=-2*y_sub_train_hat
                weight_slope=-2*np.dot(y_sub_train_hat,X_train[random_index])
                self.intercept=self.intercept-self.eta*intercept_slope
                self.weights=self.weights-self.eta*weight_slope
    def predict(self,X_test):
        Y_predicted=np.dot(X_test,self.weights)+self.intercept
        return Y_predicted
    def calculate_accuracy_score(self,Y_test,Y_predicted):
        return r2_score(Y_test,Y_predicted)

if __name__=='__main__':
    X,Y=load_diabetes(return_X_y=True)
    X_train,X_test,Y_train,Y_test=tts(X,Y,random_state=2,test_size=0.2)
    sgd=SGDregressor(epochs=50) 
    sgd.train(X_train,Y_train)
    Y_predicted=sgd.predict(X_test)
    score=sgd.calculate_accuracy_score(Y_test,Y_predicted)
    print(score)