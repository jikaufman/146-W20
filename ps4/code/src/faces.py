"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Famous Faces
"""

# python libraries
import collections

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# libraries specific to project
import util
from util import *
from cluster import *

import time

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y) :
    """
    Translate images to (labeled) points.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets
    
    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """
    
    n,d = X.shape
    
    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in xrange(n) :
        images[y[i]].append(X[i,:])
    
    points = []
    for face in images :
        count = 0
        for im in images[face] :
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """
    
    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()


def generate_points_2d(N, seed=1234) :
    """
    Generate toy dataset of 3 clusters each with N points.
    
    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed
    
    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)
    
    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]
    
    label = 0
    points = []
    for m,s in zip(mu, sigma) :
        label += 1
        for i in xrange(N) :
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))
    
    return points


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k) :
    """
    Randomly select k unique elements from points to be initial cluster centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2c: implement (hint: use np.random.choice)
    return np.random.choice(points, k, replace=False)
    ### ========== TODO : END ========== ###


def cheat_init(points) :
    """
    Initialize clusters by cheating!
    
    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2f: implement
    initial_points = []
    
    label_dict = {}
    for p in points:
        if p.label in label_dict:
            label_dict[p.label].append(p)
        else:
            label_dict[p.label] = [p]
            
    for key, val in label_dict.items():
        cluster = Cluster(val)
        initial_points.append(cluster.medoid())
    
    return initial_points
    ### ========== TODO : END ========== ###


def plot_and_update(points, k, cluster_centers, clustering_function, plot):
    clusters_list = []

    for j in range(0, k):
        clusters_list.append([])

    for point in points:
        min_i = 1000000
        min_distance = 1000000

        for i in range(len(cluster_centers)):
            current_distance = point.distance(cluster_centers[i])
            if current_distance < min_distance:
                min_i = i
                min_distance = current_distance

        clusters_list[min_i].append(point)

    temp_cluster_set = ClusterSet()

    for l in range(len(clusters_list)):
        c = Cluster(clusters_list[l])
        temp_cluster_set.add(c)

    if plot:
        plot_clusters(temp_cluster_set, "Random init", clustering_function)

    return clusters_list, temp_cluster_set

def kAverage(points, k, average, init='random', plot=False):
    k_clusters = ClusterSet()

    cluster_centers = random_init(points, k) if init == 'random' else cheat_init(points)

    clusters_list, temp_cluster_set = plot_and_update(points, k, cluster_centers, average, plot)
    old_cluster = k_clusters

    while not temp_cluster_set.equivalent(old_cluster):
        old_cluster = temp_cluster_set

        cluster_centers = average(temp_cluster_set)

        clusters_list, temp_cluster_set = plot_and_update(points, k, cluster_centers, average, plot)

        if temp_cluster_set.equivalent(old_cluster):
            return temp_cluster_set


def kMeans(points, k, init='random', plot=False) :
    """
    Cluster points into k clusters using variations of k-means algorithm.
    
    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        average -- method of ClusterSet
                   determines how to calculate average of points in cluster
                   allowable: ClusterSet.centroids, ClusterSet.medoids
        init    -- string, method of initialization
                   allowable: 
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm
    
    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """
    
    ### ========== TODO : START ========== ###
    # part 2c: implement
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to create
    #       a new ClusterSet object and update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).
    curr_centroids = []
    
    if init == 'random':
        curr_centroids = random_init(points, k)
    else:
        curr_centroids = cheat_init(points)
        
    prev_clusters = None
    iteration = 0
    while True:
        iteration += 1
        curr_clusters = ClusterSet()
        
        dic = {} # {index of centroid : [points]}
        for p in points:
            min_dist, min_index = np.inf, -1
            for index in range(len(curr_centroids)):
                if p.distance(curr_centroids[index]) < min_dist:
                    min_dist = p.distance(curr_centroids[index])
                    min_index = index
        
            if min_index in dic:
                dic[min_index].append(p)
            else:
                dic[min_index] = [p]
            
        for key, val in dic.items():
            curr_clusters.add(Cluster(val))
            
            
        if plot:
            plot_clusters(curr_clusters, 'kMeans iter: {}'.format(iteration), ClusterSet.centroids)
        
        if prev_clusters is not None and curr_clusters.equivalent(prev_clusters):
            return curr_clusters
        else:
            prev_clusters = curr_clusters
            curr_centroids = curr_clusters.centroids()
        
    return []
    ### ========== TODO : END ========== ###


def kMedoids(points, k, init='random', plot=False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO : START ========== ###
    # part 2e: implement
    curr_medoids = []
    
    if init == 'random':
        curr_medoids = random_init(points, k)
    else:
        curr_medoids = cheat_init(points)
        
    prev_clusters = None
    iteration = 0
    while True:
        iteration += 1
        curr_clusters = ClusterSet()
        
        dic = {} # {index of centroid : [points]}
        for p in points:
            min_dist, min_index = np.inf, -1
            for index in range(len(curr_medoids)):
                if p.distance(curr_medoids[index]) < min_dist:
                    min_dist = p.distance(curr_medoids[index])
                    min_index = index
        
            if min_index in dic:
                dic[min_index].append(p)
            else:
                dic[min_index] = [p]
            
        for key, val in dic.items():
            curr_clusters.add(Cluster(val))
            
            
        if plot:
            plot_clusters(curr_clusters, 'kMedoids iter: {}'.format(iteration), ClusterSet.medoids)
        
        if prev_clusters is not None and curr_clusters.equivalent(prev_clusters):
            return curr_clusters
        else:
            prev_clusters = curr_clusters
            curr_medoids = curr_clusters.medoids()
        
    return []
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################

def main() :
    ### ========== TODO : START ========== ###
    # part 1: explore LFW data set
    X, y = get_lfw_data()

    #show_image(im=X[2])
    #show_image(im=X[1])
    #show_image(im=X[3])

    #average_image = np.mean(X, axis=0)
    #show_image(im=average_image)

    U, mu = PCA(X)
    #plot_gallery([vec_to_image(U[:, i]) for i in xrange(12)])


    # Selecting the dimension, l, to map all features to
    for l in [1, 10, 50, 100, 500, 1288]:
        Z, Ul = apply_PCA_from_Eig(X, U, l, mu)
        X_rec = reconstruct_from_PCA(Z, Ul, mu)
        #plot_gallery([vec_to_image(X_rec[i]) for i in xrange(12)], subtitles=["l="+str(l)+",n="+str(j) for j in xrange(12)])

    # Original 12 Images
    #plot_gallery([vec_to_image(X[i]) for i in xrange(12)])

    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part 2d: cluster toy dataset
    #np.random.seed(1234)
    #points = generate_points_2d(20)
    #kMeans(points, 3, init='cheat', plot=True)
    #kMedoids(points, 3, init='cheat', plot=True)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###    
    # part 3a: cluster faces
    np.random.seed(1234)
    X1, y1 = limit_pics(X, y, [4, 6, 13, 16], 40)
    points = build_face_image_points(X1, y1)

    k_means_scores = []    
    k_medoids_scores = []

    #for _ in range(10):
        #clusters = kMeans(points, 4, init='random', plot=False)
        #k_means_scores.append(clusters.score())
        #clusters = kMedoids(points, 4, init='random', plot=False)
        #k_medoids_scores.append(clusters.score())


    #print('k-means average: {}'.format(np.mean(k_means_scores)))
    #print('k-means min: {}'.format(np.min(k_means_scores)))
    #print('k-means max: {}'.format(np.max(k_means_scores)))
    #print('k-medoids average: {}'.format(np.mean(k_medoids_scores)))
    #print('k-medoids min: {}'.format(np.min(k_medoids_scores)))
    #print('k-medoids max: {}'.format(np.max(k_medoids_scores)))
        
    # part 3b: explore effect of lower-dimensional representations on clustering performance
    np.random.seed(1234)
    X2, y2 = util.limit_pics(X, y, [4, 13], 40)
    #
    kmeans_scores_dict = dict()
    kmedoids_scores_dict = dict()
    #
    for l in np.arange(1, 42):
        Z2, Ul2 = apply_PCA_from_Eig(X2, U, l, mu)
        X_rec2 = reconstruct_from_PCA(Z2, Ul2, mu)
        points = build_face_image_points(X_rec2, y2)
    #
        cluster_set1 = kMeans(points, 2, "cheat")
        cluster_set2 = kMedoids(points, 2, "cheat")
    #
        kmeans_scores_dict[l] = cluster_set1.score()
        kmedoids_scores_dict[l] = cluster_set2.score()
    #
    plt.plot(list(kmeans_scores_dict.keys()), list(kmeans_scores_dict.values()), 'r', label='K-means')
    plt.plot(list(kmedoids_scores_dict.keys()), list(kmedoids_scores_dict.values()), 'b', label='K-medoids')
    plt.title('Score for kMeans and kMedoids vs. # Principal Components')
    plt.xlabel('# Principal Components')
    plt.ylabel('score')
    plt.legend()
    #plt.show()
    
    # part 3c: determine ``most discriminative'' and ``least discriminative'' pairs of images
    np.random.seed(1234)
    max_score = (-1, None, None)
    min_score = (np.Inf, None, None)

    for i in np.arange(0, 19):
        for j in np.arange(0, 19):
            if i != j:
                X_ij, y_ij = util.limit_pics(X, y, [i, j], 40)
                points = build_face_image_points(X_ij, y_ij)
                cluster_set = kMedoids(points, 2, init='cheat')
                score = cluster_set.score()
                if score < min_score[0]:
                    min_score = (score, i, j)
                if score > max_score[0]:
                    max_score = (score, i, j)

    print max_score
    print min_score
    plot_representative_images(X, y, [min_score[1], min_score[2]], title='min score images')
    plot_representative_images(X, y, [max_score[1], max_score[2]], title='max score images')
    
    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
