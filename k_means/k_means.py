import numpy as np 
import pandas as pd
import random
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, k = 2, distortion_threshold = 8.8, silhouette_threshold = 0.67, max_iterations = 50, mean=True):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.k = k
        self.distortion_threshold = distortion_threshold
        self.silhouette_threshold = silhouette_threshold
        self.max_iterations = max_iterations
        self.mean = mean
        
    def fit(self, X):
        """
        Estimates parameters for the classifier until the euclidean 
        distortion threshold is met.
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        centroids = initialize_centroids(X, self.k)
        
        # Initialize random cluster centroids
        for i in range(self.max_iterations):
            z = centroid_assignment_indices(X, centroids, self.k)
            distortion = euclidean_distortion(X, z)
            silhouette = euclidean_silhouette(X, z)
            new_centroids = self.update_centroids(X, z)
            # We break before max iterations is reached if the thresholds
            # are fulfilled, or if no centroids are updated.
            if((distortion < self.distortion_threshold and silhouette > self.silhouette_threshold) or np.all(centroids == new_centroids)):
               break
            
            centroids = new_centroids
        self.centroids = centroids
       
               
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        z = centroid_assignment_indices(X, self.centroids, self.k)
        distortion = euclidean_distortion(X, z)
        silhouette = euclidean_silhouette(X, z)
        return z
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids
    
    def update_centroids(self, X, z):
        new_centroids = np.zeros((self.k, X.shape[1]))
        if(self.mean):
            # Calculate the mean of all data points assigned to this cluster
            for cluster in range(self.k):
                cluster_points = X[np.asarray(z) == cluster]
                if len(cluster_points) > 0:
                    new_centroids[cluster] = np.mean(cluster_points, axis=0)
        else: 
            # USES MEDIAN INSTEAD OF MEAN
            for cluster in range(self.k):
                cluster_points = X[np.asarray(z) == cluster]
                if len(cluster_points) > 0:
                    new_centroids[cluster] = np.median(cluster_points, axis=0)
        return new_centroids
    
    def normalize(self, X):
        """
        Perform min-max scaling on the input matrix X.

        Args:
            X (numpy.ndarray): Input data with shape (m, n), where m is the number of samples, and n is 
            the number of features.

        Returns:
            numpy.ndarray: Scaled data with the same shape as X.
        """
        # Compute the minimum and maximum values for each feature (column)
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)

        # Perform min-max scaling
        scaled_X = (X - min_vals) / (max_vals - min_vals)

        return scaled_X
       
# --- Some utility functions 


def initialize_centroids(X, k):
    # returns k random data points from within X
    centroids = []
    for i in range(k):
        random_cluster_centre = X.iloc[random.randint(0, (X.shape[0]))]
        centroids.append(random_cluster_centre)
    return centroids

def centroid_assignment(X, centroids, k):
    '''
    How the data points are assigned to each centroid is not intuitive. 
    This method returns an array of k arrays each corresponding to individual clusters 
    and containing the datapoints assigned to them.
    
    This method is UNUSED.
    '''
    raise NotImplemented
    centroid_assignment_array = [[] for _ in range(k)]
    for i in range (X.shape[0]):
        x_i = X.iloc[i]
        distances = []
        for centroid_j in centroids:
            distances.append(euclidean_distance(x_i, centroid_j))
        closest_centroid_index = np.argmin(distances)
        centroid_assignment_array[closest_centroid_index].append(x_i)
    return centroid_assignment_array

def centroid_assignment_indices(X, centroids, k):
    centroid_assignment_index_array = []
    for i in range (X.shape[0]):
        x_i = X.iloc[i]
        distances = []
        for centroid_j in centroids:
            distance = euclidean_distance(x_i, centroid_j)
            distances.append(distance)
        closest_centroid_index = np.argmin(distances)
        centroid_assignment_index_array.append(closest_centroid_index)
    return centroid_assignment_index_array



def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion.
    
    Info: The distortion function J is a non-convex function, and so coordinate
        descent on J is not guaranteed to converge to the global minimum
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()#(axis=1)
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  