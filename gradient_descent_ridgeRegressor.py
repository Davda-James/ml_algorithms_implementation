import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split as tts 

class Ridgeregressor:
    def __init__(self,learning_rate=0.01,epochs=100,alpha=0.185):
        self.alpha=alpha
        self.eta=learning_rate
        self.epochs=epochs
        self.weights =None
        self.intercept=None 

    def train(self,X_train,Y_train):
        X_train=np.insert(X_train,0,1,axis=1)
        allweights=np.ones(X_train.shape[1])
        for i in range(self.epochs):
            weight_slope=2*(self.alpha*self.weights + np.dot(np.dot(X_train.T,X_train),self.weights)-np.dot(X_train.T,Y_train))
            self.allweights=self.allweights-self.eta*weight_slope
        self.intercept=allweights[0]
        self.weights=allweights[1:]
    def test(self,X_test):
        return np.dot(X_test,self.weights)+self.intercept

    def calculate_score(self,Y_test,Y_predicted):
        r2_score(Y_test,Y_predicted)


if __name__=='__main__':
    X,Y=load_diabetes(return_X_y=True)
    X_train,X_test,Y_train,Y_test =tts(X,Y,random_state=2,test_size=0.2)
    rg =Ridgeregressor()
    rg.train(X_train,Y_train)
    Y_predicted=rg.test(X_test)
    score=rg.calculate_score(Y_test,Y_predicted)
    print(rg.weights,rg.intercept)
    print(score)


