import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import random 

class KMeans:
    def __init__(self,n_clusters=2,max_iter=300):
        self.n_clusters=n_clusters
        self.max_iter=max_iter
        self.centroids=None 

    def fit_predict(self,x):
        idx=random.sample(range(0,x.shape[0]),self.n_clusters)
        self.centroids=x[idx]
        for i in range(self.max_iter):
            cluster=self.assign_clusters(x)
            new_centroids=self.update_centroids(x,cluster)
            if np.all(self.centroids==new_centroids):
                break
            self.centroids=new_centroids
        return cluster

    def update_centroids(self,x,clusters):
        new_centroids=np.zeros((self.n_clusters,x.shape[1]))
        for i in range(self.n_clusters):
            new_centroids[i]=np.mean(x[clusters==i],axis=0)
        return new_centroids

    def assign_clusters(self,x):
        cluster=np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            distances=np.linalg.norm(x[i]-self.centroids,axis=1)
            cluster[i]=np.argmin(distances)
        return cluster

if __name__=='__main__':
    x=np.array([[1.5, 2.3],[2.1, 1.8],[1.9, 2.4],[2.3, 1.9],[2.0, 2.5],[7.2, 6.9],[6.8, 7.4],[7.3, 7.1],[7.0, 7.6],[6.9, 6.8],[3.8, 9.1],[4.1, 8.9],[4.0, 9.3],[3.7, 9.0],[4.3, 9.2]])
    km=KMeans(3,100)
    cluster=km.fit_predict(x)
    print(cluster)
    plt.scatter(x[:,0], x[:,1], c=cluster, s=80)
    plt.show()