import numpy  as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as tts 
from sklearn.metrics import r2_score,accuracy_score
import matplotlib.pyplot as plt 
class logisticRegressor:
    def __init__(self,learning_rate=0.01,epochs=100):
        self.epochs=epochs
        self.eta=learning_rate 
        self.intercept_=None 
        self.coeff_=None
        self.weights=None 
    def __helper(self,y):
        if y>0:
            return 1
        return 0
    def train(self,xtrain,ytrain):
        xtrain_new=np.insert(xtrain,0,1,axis=1)
        self.weights=np.ones(xtrain_new.shape[1])
        for i in range(self.epochs):
            index=np.random.randint(0,xtrain_new.shape[0])
            y_hat=np.dot(xtrain_new[index,],self.weights)
            y_hat_pred=self.__helper(y_hat)
            self.weights=self.weights+(ytrain[index]-y_hat_pred)*self.eta*xtrain_new[index]
        self.intercept_=self.weights[0]
        self.coeff_ =self.weights[1:]

    def predict(self,xtest):
        return np.where(np.dot(xtest,self.coeff_)+self.intercept_>0,1,0)
        
if __name__=="__main__":
    x,y=make_classification(n_samples=300,n_features=2,n_informative=1,n_redundant=0,n_classes=2,n_clusters_per_class=1,class_sep=1,random_state=80)
    xtrain,xtest,ytrain,ytest=tts(x,y,random_state=2,test_size=0.3)
    lr=logisticRegressor(epochs=1000,learning_rate=0.2)
    lr.train(xtrain,ytrain)
    y_predicted=lr.predict(xtest)
    score=accuracy_score(ytest,y_predicted)
    print(score)
    plt.figure(figsize=(10,6))
    plt.xlim(x[:, 0].min() - 1, x[:, 0].max() + 1)  # Add padding around the min and max values
    plt.ylim(x[:, 1].min() - 1, x[:, 1].max() + 1)
    plt.xticks(ticks=np.linspace(x[:, 0].min(), x[:, 0].max(), num=10))  # Adjust tick intervals
    plt.yticks(ticks=np.linspace(x[:, 1].min(), x[:, 1].max(), num=10))
    plt.scatter(x[:,0],x[:,1],c=y,cmap='winter',s=100)
    xpoints=np.linspace(-50,50,30)
    m=-lr.coeff_[0]/lr.coeff_[1]
    c=-lr.intercept_/lr.coeff_[1]
    plt.plot(xpoints,m*xpoints+c,color='red')
    plt.tight_layout()
    plt.show()