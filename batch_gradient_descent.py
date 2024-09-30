from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split as tts 
import numpy as np 
from sklearn.metrics import r2_score
class batch_gradient_regressor:
    def __init__(self,epochs=100,learning_rate=0.01):
        self.epochs=epochs
        self.eta=learning_rate
        self.b=None 
        # @users b is for intercept 
        self.weights=None
        # @users weights represent the coefficient of the respective features 
    def train(self,X_train,Y_train):
        self.b=0
        self.weights=np.ones(X_train.shape[1])
        for i in range(self.epochs):
            Y_hat=np.dot(X_train,self.weights)+self.b
            Y_train_sub_hat=Y_train-Y_hat
            b_slope=-2*np.mean(Y_train_sub_hat)
            weight_slope=(-2*np.dot(Y_train_sub_hat,X_train))/X_train.shape[0]
            self.b=self.b-self.eta*b_slope
            self.weights=self.weights-self.eta*weight_slope
    def predict(self,X_test,Y_test):
        y_predicted=np.dot(X_test,self.weights)+self.b
        return y_predicted
if __name__=='__main__': 
    # this is just the usage of the implemented class score may vary as I have not fine tuned it 
    X,Y=load_diabetes(return_X_y=True)
    X_train,X_test,Y_train,Y_test=tts(X,Y,test_size=0.2,random_state=2)
    bgr=batch_gradient_regressor(epochs=1000,learning_rate=0.2)
    bgr.train(X_train,Y_train)
    Y_predicted=bgr.predict(X_test,Y_test)
    score=r2_score(Y_test,Y_predicted)
    print(score)