import numpy as np 
from sklearn.datasets import load_diabetes 
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
class ridgeRegressor:
    def __init__(self,alpha=-0.1):
        self.intercept=None 
        self.weight=None    
        self.alpha=alpha
    
    def train(self,X_train,Y_train):
        X_mean=X_train.mean();
        Y_mean=Y_train.mean();
        num=0
        den=self.alpha
        for i in range(X_train.shape[0]):
            num = num + (Y_train[i]-Y_mean)*(X_train[i]-X_mean)
            den = den + (X_train[i]-X_mean)**2;
        self.weight=num/den
        self.intercept=Y_mean-self.weight*X_mean
    def predict(self,X_test):
        Y_predicted=self.weight*X_test+self.intercept
        return Y_predicted

    
    def accuracy_score(self,Y_test,Y_predicted):
        score=r2_score(Y_test,Y_predicted);
        return score;



if __name__=='__main__':
    X,Y=make_regression(n_samples=100,n_features=1,n_informative=1,n_targets=1,noise=30,random_state=2)
    X_train,X_test,Y_train,Y_test=tts(X,Y,random_state=2,test_size=0.2);
    rr=Ridge(alpha=0.1)
    rr.fit(X_train,Y_train)
    rrg=ridgeRegressor()
    rrg.train(X,Y)
    Y_predicted=rrg.predict(X_test)
    plt.scatter(X,Y);
    x_plot=np.arange(-4,5)
    plt.plot(x_plot,rr.intercept_+x_plot*rr.coef_[0],color="orange")
    plt.plot(x_plot,rrg.intercept+x_plot*rrg.weight,color="red")
    plt.show()
