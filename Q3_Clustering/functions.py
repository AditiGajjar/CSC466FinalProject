import pandas as pd
import sys
import numpy as np


def calc_dist(p1, p2, euclid = True, ax = 0):
    if euclid:
        dist = euclidean(p1, p2, ax)
    else:
        dist = manhattan(p1, p2, ax)
    return dist


def calc_dist_mat(mat, euclid):
    rows = mat.shape[0]
    dist = np.zeros((rows,rows))
    for i in range(rows):
        for j in range(i+1,rows):
            dist[i,j] = calc_dist(mat[i], mat[j], euclid)
    dist = dist + dist.T
    return dist

def array_in_list(list_of_arrays, array_to_find):
    for arr in list_of_arrays:
        if np.array_equal(arr, array_to_find):
            return True
    return False

def encode_df(D, norm = False): #convert categ to numeric and normalize numeric!!!!!!!!!!!
    # Convert all columns to float
    for col in D.columns: #should use .apply() but life goes on
        D[col] = pd.to_numeric(D[col], errors='coerce')
        if norm:
            D[col] = min_max_scaling(D[col])
    D = D.to_numpy()
    return D

"""
Return dictionary of metrics for each cluster
Want outs =[(numpts, value of center, (max, min, avrg distances), sse),....]
"""
def metrics(clusters, euclid): #[(cluster1, [pt1, pt2,..]), ...] such that all points and clusterss are np arrays of the observed values
    outs = []
    for index, clust in enumerate(clusters):
        clust_cent = clust[0] #want np array of form (x1, x2,.. x_n)
        points = clust[1]
        indexes = clust[2]
        dists = []
        sse = 0
        for point in points:
            dist = calc_dist(point, clust_cent, euclid)
            dists.append(dist)

            sse += euclidean(clust_cent,point)**2

        dists = np.array(dists)
        max_dist = np.max(dists)
        min_dist = np.min(dists)
        mean_dist = np.mean(dists)


        dist_mets = [max_dist, min_dist, mean_dist]
        out = [index, clust_cent, dist_mets, sse, points, indexes]
        outs.append(out)
    return outs

def format_met(metrics):
    total_out = []
    for cluster in metrics:
        l1 = "Cluster {}:".format(cluster[0])
        l2 = "Center {}".format(np.array2string(cluster[1], separator= ","))
        l3 = "Max Distance to Center: {}".format(cluster[2][0])
        l4 = "Min Distance to Center: {}".format(cluster[2][1])
        l5 = "Average Distance to Center: {}".format(cluster[2][2])
        l6 = "SSE from Center: {}".format(cluster[3])
        l7 = "{} Points:".format(len(cluster[4]))
        outputs = [l1,l2,l3,l4,l5,l6,l7]
        # for i, point in enumerate(cluster[4]):
        #     outputs.append(str(cluster[5][i]) + " - " + np.array2string(point, separator= ","))
        outputs.append("Top genres: " + str(pd.Series(cluster[5]).mode()[0]))
        outputs.append("-----------------------------------------------------------\n")
        total_out.append(outputs)
    return total_out
    

def output_mets(mets):
    for clust in mets:
        for out in clust:
            print(out)

def test_metrics(mets):
    sse = 0
    for met in mets:
        sse += met[3]
    return sse

def euclidean(n1, n2, ax = 0):
    return np.linalg.norm(n1 - n2, axis= ax) 

def manhattan(n1, n2, ax = 0):
    return np.linalg.norm(n1 - n2, ord=1, axis = ax) 

def cosine(n1, n2):
    return np.dot(n1,n2) / (np.linalg.norm(n1)*np.linalg.norm(n2))

def min_max_scaling(col):
    min_val = col.min()
    max_val = col.max()
    scaled_column = (col - min_val) / (max_val - min_val)
    return scaled_column

def parser(path):
    ground_truth = None
    spotify_df = pd.read_csv(path)
    ground_truth = spotify_df['playlist_genre']
    spotify_numeric = spotify_df.select_dtypes(include=['number'])

    return spotify_numeric, ground_truth

def parser_DBScan(path):
    ground_truth = None
    spotify_df = pd.read_csv(path).sample(1000)
    ground_truth = spotify_df['playlist_genre']
    spotify_numeric = spotify_df.select_dtypes(include=['number'])

    return spotify_numeric, ground_truth