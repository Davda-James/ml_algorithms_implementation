import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split as tts 

class Ridgeregressor:
    def __init__(self,alpha=0.01):
        self.alpha=alpha
        self.weights =None
        self.intercept=None 

    def train(self,X_train,Y_train):
        identity_mtx=np.identity(X_train.shape[1]+1,dtype=float)
        X_train=np.hstack((np.ones((X_train.shape[0],1),dtype=float),X_train))
        Xt_X =np.dot(X_train.T,X_train)
        Xt_y=np.dot(X_train.T,Y_train)
        mid_mtx=np.linalg.inv(Xt_X+self.alpha*identity_mtx)
        final_mtx=np.dot(mid_mtx,Xt_y)
        self.intercept=final_mtx[0]
        self.weights=final_mtx[1:]
    def test(self,X_test):
        return np.dot(X_test,self.weights)+self.intercept 

    def calculate_score(self,Y_test,Y_predicted):
        return r2_score(Y_test,Y_predicted)

if __name__=='__main__':
    X,Y=load_diabetes(return_X_y=True)
    X_train,X_test,Y_train,Y_test =tts(X,Y,random_state=2,test_size=0.2)
    rg =Ridgeregressor()
    rg.train(X_train,Y_train)
    Y_predicted=rg.test(X_test)
    score=rg.calculate_score(Y_test,Y_predicted)
    print(rg.weights,rg.intercept)


