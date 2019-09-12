import numpy as np

class KNearestNeighbors():
    def __init__(self, X_train, y_train, n_neighbors=5, weights = 'uniform'):
    
        self.X_train = X_train
        self.y_train = y_train
        
        self.n_neighbors = n_neighbors
        self.weights = weights
        
        self.class_num = 3

        
    def euclidian_distance(self, a, b):     
        return np.sqrt(np.sum((a-b)**2, axis=1))
    
    
    
    def kneighbors(self, X_test, return_distance=False):
       
        dist = []
        neigh_ind = []
        
        point_dist = [self.euclidian_distance(x_test, self.X_train) for x_test in X_test]

        for row in point_dist:
            enum_neigh = enumerate(row)
            sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:self.n_neighbors]
    
            ind_list = [tup[0] for tup in sorted_neigh]
            dist_list = [tup[1] for tup in sorted_neigh]
    
            dist.append(dist_list)
            neigh_ind.append(ind_list)
        
        if return_distance:
            return np.array(dist), np.array(neigh_ind)
        
        return np.array(neigh_ind)
         
    
    def predict(self, X_test):
        
        if self.weights=='uniform':
            neighbors = self.kneighbors(X_test)
            y_pred = np.array([np.argmax(np.bincount(self.y_train[neighbor])) for neighbor in neighbors])
        
            return y_pred 
    
        if self.weights=='distance':
        
            dist, neigh_ind = self.kneighbors(X_test, return_distance=True)
        
            inv_dist = 1/dist
            
            mean_inv_dist = inv_dist / np.sum(inv_dist, axis=1)[:, np.newaxis]
            
            proba = []
            
            for i, row in enumerate(mean_inv_dist):
                
                row_pred = self.y_train[neigh_ind[i]]
                
                for k in range(self.class_num):
                    indices = np.where(row_pred==k)
                    prob_ind = np.sum(row[indices])
                    proba.append(np.array(prob_ind))
        
            predict_proba = np.array(proba).reshape(X_test.shape[0], self.class_num)
            
            y_pred = np.array([np.argmax(item) for item in predict_proba])
            
            return y_pred
            
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        return float(sum(y_pred == y_test))/ float(len(y_test))