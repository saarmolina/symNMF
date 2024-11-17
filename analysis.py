import sys
import math
import numpy as np
from sklearn.metrics import silhouette_score
import symnmf

# Global variables for k-means parameters
EPSILON = 0.001
DEFAULT_ITERATIONS = 200
MAX_ITER = 1000

def eucDistance(vector1, vector2):
    """
    Calculate the Euclidean distance between two vectors.
    """
    dis = 0 
    d = len(vector1)
    for i in range(d):
        dis += (vector1[i]-vector2[i])**2
    return math.sqrt(dis)

def assignToClusters(centroids, vectors):
    """
    Assign each vector to the closest centroid.
    """
    k = len(centroids)
    clusters = [[] for i in range(k)]
    
    for vector in vectors:
        minDistance = float('inf')
        closestCentroid = -1 
        
        for i in range(k):
            centroid = centroids[i]
            currDistance = eucDistance(vector, centroid)
            
            if currDistance < minDistance:
                minDistance = currDistance 
                closestCentroid = i
                
        clusters[closestCentroid].append(vector)
    
    return clusters

def newCentroid(cluster):
    """
    Calculate the new centroid of a cluster.
    """
    d = len(cluster[0])
    n = len(cluster)
    newCentroid = [0 for i in range(d)]
    
    for vector in cluster:
        for i in range(d):
            newCentroid[i] += vector[i]
    
    for i in range(d):
        newCentroid[i] = newCentroid[i]/n
        
    return newCentroid

def updateCentroids(clusters):
    """
    Update centroids based on current clusters.
    """
    k = len(clusters)
    centroids = [-1 for i in range(k)]
    
    for i in range(k):
        cluster = clusters[i]
        if len(cluster) == 0:
            continue
        newCentroidVal = newCentroid(cluster)
        centroids[i] = newCentroidVal
    
    return centroids

def finishIter(oldCentroids, centroids, EPSILON):
    """
    Check if the centroids have converged.
    """
    k = len(centroids)
    
    for i in range(k):
        distance = eucDistance(oldCentroids[i], centroids[i])
        if distance >= EPSILON:
            return False
            
    return True

def kmeans(k, iter, vectors):
    """
    k-means clustering algorithm.
    """
    centroids = vectors[:k]
    
    for i in range(iter):
        oldCentroids = centroids
        clusters = assignToClusters(centroids, vectors)
        centroids = updateCentroids(clusters)
        
        if finishIter(oldCentroids, centroids, EPSILON):
            break
            
    return centroids

def get_clusters_from_H(H):
    """
    Assign clusters based on the largest value in each row of H.
    """
    clusters = np.argmax(H, axis=1)
    if len(np.unique(clusters)) < 2:
        # If all points assigned to same cluster, reassign second highest to another cluster
        second_best = np.argsort(H, axis=1)[:, -2]
        idx = np.argmax(H[np.arange(len(H)), second_best])
        clusters[idx] = 1
    return clusters


def get_kmeans_clusters(centroids, vectors):
    """
    Convert kmeans output to cluster assignments format
    Returns an array where index i contains the cluster number for point i
    """
    labels = np.zeros(len(vectors))
    for i, vector in enumerate(vectors):
        min_dist = float('inf')
        closest_centroid = -1
        for j, centroid in enumerate(centroids):
            dist = eucDistance(vector, centroid)
            if dist < min_dist:
                min_dist = dist
                closest_centroid = j
        labels[i] = closest_centroid
    return labels

def read_data(file_name):
    """Read and validate input data"""
    try:
        data = np.loadtxt(file_name, delimiter=',')
        # Handle 1D case by reshaping to 2D array
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        return data, data.tolist()
    except:
        print("An Error Has Occurred")
        sys.exit(1)

def calculate_symnmf(vectors, k, n):
    """Perform symNMF clustering"""
    W = symnmf.norm(vectors)
    if W is None:
        print("An Error Has Occurred")
        sys.exit(1)
        
    np.random.seed(1234)
    m = np.mean(W)
    H = np.random.uniform(0, 2 * np.sqrt(m/k), size=(n, k)).tolist()
    
    H_final = symnmf.symnmf(W, H, n, k)
    if H_final is None:
        print("An Error Has Occurred")
        sys.exit(1)
        
    return np.array(H_final)

def calculate_scores(data, H_final, vectors, k):
    """Calculate silhouette scores for both methods"""
    try:
        nmf_clusters = get_clusters_from_H(H_final)
        nmf_score = silhouette_score(data, nmf_clusters)
        centroids = kmeans(k, DEFAULT_ITERATIONS, vectors)
        kmeans_clusters = get_kmeans_clusters(centroids, vectors)
        kmeans_score = silhouette_score(data, kmeans_clusters)
        
        return nmf_score, kmeans_score
    except:
        print("An Error Has Occurred2")
        sys.exit(1)

def main():
    """Main function handling the analysis flow"""
    try:
        k = int(sys.argv[1])
        file_name = sys.argv[2]
    except:
        print("An Error Has Occurred")
        sys.exit(1)
    
    data, vectors = read_data(file_name)
    H_final = calculate_symnmf(vectors, k, len(data))
    nmf_score, kmeans_score = calculate_scores(data, H_final, vectors, k)
    
    print(f"nmf: {nmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")

if __name__ == "__main__":
    main()