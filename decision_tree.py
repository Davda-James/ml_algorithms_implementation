import numpy as np 
import pandas as pd 
class TreeNode:
    def __init__(self,feature=None,threshold=None,left=None,right=None,value=None):
        self.feature=feature 
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

class decisionTree:
    def __init__(self,max_depth=None):
        self.max_depth=max_depth
    def fit(self,x,y):
        self.tree=self.build_tree(np.column_stack((x,y)))
    def build_tree(self,data,curr_depth=0):
        if curr_depth>=self.max_depth:
            return TreeNode(value=self.majority_class(data))
        if len(set(data[:,-1]))==1:
            return TreeNode(value=data[0,-1]) 
        best_feature,best_threshold= self.find_best_split(data)
        if best_feature is None:
            return TreeNode(value=self.majority_class(data))
        left_data,right_data=self.split_data(data,best_feature,best_threshold)
        left_child=self.build_tree(left_data,curr_depth+1)
        right_child=self.build_tree(right_data,curr_depth+1)
        return TreeNode(feature=best_feature,threshold=best_threshold,left=left_child,right=right_child)            
    def find_best_split(self,data):
        best_gain=-1
        best_threshold=None 
        best_feature=None 
        num_of_features=data.shape[1]-1
        for feature in range(num_of_features):
            thresholds=np.unique(data[:,feature])
            for threshold in thresholds:
                gain=self.information_gain(data,feature,threshold)
                if gain> best_gain:
                    best_gain=gain
                    best_feature=feature 
                    best_threshold=threshold
        return best_feature,best_threshold
    
    def information_gain(self,data,feature,threshold):
        left_data,right_data=self.split_data(data,feature,threshold)
        parent_impurity=self.gini_index(data)
        left_data_impurity=self.gini_index(left_data)
        right_data_impurity=self.gini_index(right_data)
        weighted_impurity=(len(left_data)/len(data))*left_data_impurity+(len(right_data)/len(right_data))*right_data_impurity             
        return parent_impurity-weighted_impurity
    def split_data(self,data,feature,threshold):
        if isinstance(threshold,(int,float)):
            left=data[data[:,feature]<threshold]
            right=data[data[:,feature]>=threshold]
        else:
            left=data[data[:,feature]==threshold]
            right=data[data[:,feature]!=threshold]
        return left,right
    def gini_index(self,data):
        labels,count=np.unique(data[:,-1],return_counts=True)
        labels_prob=count/len(data)
        return 1-np.sum(labels_prob**2)
    def majority_class(self,data):
        # return np.bincount(data[:,-1]).argmax()
        labels, counts = np.unique(data[:, -1], return_counts=True)
        return labels[np.argmax(counts)]
    def predict(self,x):
        predictions=[self._predict(input_data,self.tree) for input_data in x]
        return np.array(predictions)
    def _predict(self,input_data,node):
        if node.value is not None:
            return node.value 
        if isinstance(node.threshold,(int,float)):
            if input_data[node.feature] < node.threshold:
                return self._predict(input_data,node.left)
            else:
                return self._predict(input_data,node.right)
        else:
            if input_data[node.feature]==node.threshold:
                return self._predict(input_data,node.left)
            else:
                return self._predict(input_data,node.right)
            
            
if __name__=='__main__':
    tree=decisionTree(max_depth=2)
    train_data = pd.DataFrame({
        'Outlook_Sunny': [1, 0, 0, 0, 0],
        'Outlook_Overcast': [0, 1, 0, 0, 1],
        'Outlook_Rainy': [0, 0, 1, 1, 0],
        'Temperature': [85, 80, 83, 70, 65],
        'Humidity': [85, 70, 78, 90, 95],
        'Windy': [0, 0, 1, 1, 0],
        'Play': ['No', 'Yes', 'No', 'No', 'No']
    })
    X_train = train_data.drop('Play', axis=1).values
    y_train = train_data['Play'].values
    
    test_data = pd.DataFrame({
    'Outlook_Sunny': [1, 0, 0, 0], 
    'Outlook_Overcast': [0, 1, 0, 0],
    'Outlook_Rainy': [0, 0, 1, 1],
    'Temperature': [85, 80, 83, 70],
    'Humidity': [85, 70, 78, 90],
    'Windy': [False, False, True, True]
    })
    test_data['Windy'] = test_data['Windy'].astype(int)
    X_test = test_data.values
    actual_labels = ['No', 'Yes', 'Yes', 'No'] 
    tree.fit(X_train,y_train)
    pred=tree.predict(X_test)
    print("Predictions for test data:", pred)
    accuracy = np.sum(pred == actual_labels) / len(actual_labels)
    print("Accuracy:", accuracy)