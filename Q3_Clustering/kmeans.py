#python3 kmeans. <Filename> <K> [plus] [man] [norm]
from functions import *


"""
Picks K-random observation, returns them as cluster centers. take in nxm matrix, return list or np.array of k observations (each of length m)
"""
def random_cent(D, k):
    if D.shape[0] >= k:
        indices = np.random.choice(D.shape[0], k, replace=False)
        return D[indices]
    
    else:
        print("K too big!")
        quit

"""
Refer to google doc, essentially fancy probability way of getting decent clusters
#want to return 2 lists/arrays assignment np. 
1. array such that index represents observation (index) and value is index of cluster 
2. cluster_pts such that index is unique cluster and value is list of observation (values!)

"""
def kmeans_plus(D_in, k, euclid):
    D = D_in.copy()
    center = np.mean(D, axis=0)
    norms = calc_dist(D, center, euclid, 1)
    clusts = [D[np.argmax(norms)]]
    D = np.delete(D, np.argmax(norms), axis=0)
    while len(clusts) < k:
        numrows = D.shape[0]
        dists = np.zeros(numrows)        
        for c in clusts:
            dists += calc_dist(D, c, euclid, 1)
        far_clust = D[np.argmax(dists)] 
        if not array_in_list(clusts, far_clust):
            clusts.append(far_clust)
        D = np.delete(D, np.argmax(dists), axis=0)

    return clusts    

"""
Helper function, returns index of nearest cluster from clusters
"""
def find_nearest_clust(obs, clusters, euclid):
    dists = [] #empty numpy array of length k, only k clusters so should work 
    for cluster in clusters: #clusters = (c1, c2..,cn) such that each ci = (n1, n2,..nk) where we have k attributes
        dist = calc_dist(cluster, obs, euclid)
        
        dists.append(dist)
    nearest_ind = int(np.argmin(dists))
    return nearest_ind

def remove_array(array_rem, array_list):
    ret = [arr for arr in array_list if not np.array_equal(arr, array_rem)]
    return ret

# Helper functio due to nuance of .remove for np array
def remove_obs(rowid, cluster):
    ret = [obs for obs in cluster if not obs[2] == rowid]
    return ret


"""
Going to do numpy heavy approach for performance boost and desire to become familiar with matrix operatios over pandas
random_cent default with rand = false \implies kmeans_plus methods
similar for euclid s.t. euclid = false \implies manhattan dist
"""
def kmeans(D, k, rand = True, euclid = True, max_iter = 10000, rowids = None):
    numrows = D.shape[0]
    numcols = D.shape[1]
    if rand:
        clusters = random_cent(D, k)
    else:
        clusters = kmeans_plus(D, k, euclid)#by nature of kmeans_plus, read documentation
 #will need to create foo and cluster_pts in kmeans plus
    #want clusters to be array of arrays [c1, c2...] st c_i = [y1, y2, ...]

    cluster_pts = [[] for _ in range(k)]
 #WaNT A NESTED LIST SUCH THAT cluster_pts[i] is all the current observations assigned to cluster i
    if rowids is None:
        rowids = [i for i in range(numrows)]
    #initilaizing step
    assigned_clust = [] #index represents observation, value is index of cluster  assigned
    """
    Concept below is to iterate over each observation
    then each cluster
    find nearest cluster and store the index of that cluster from the 'total clusters' list
    """
    
    for index, obs in enumerate(D): #obs is row of matrix
        nearest_ind = find_nearest_clust(obs, clusters, euclid)
        assigned_clust.append(nearest_ind) #a
        rowid = rowids[index]
        cluster_pts[nearest_ind].append((obs, rowid, index)) #will be a nested list of numpy arrays

    #updating step
    change = True
    i = 0
    while change and i < max_iter: #should terminate when no reassignments happen
        change = False #assume no change has happened, then when it does cha
        new_clusts = np.empty((k, numcols))
        for index, points in enumerate(cluster_pts): #points is list of numpy arrays
            observations = []
            for point in points:
                observations.append(point[0])
            if observations: #not empty
                mean_values = np.mean(observations, axis=0) 
                new_clusts[index] = mean_values
            else:
                print("ISOLATED CLUSTER!")
        #calculates mean of current clusters
        for ind, obs in enumerate(D):
            nearest_ind = find_nearest_clust(obs, new_clusts, euclid)
            curr_clust = int(assigned_clust[ind])
            if nearest_ind != curr_clust: #if closest cluster is not the one it is already assigned to
                change = True
                 #gets index of old cluster
                assigned_clust[ind] = nearest_ind #assigns new closer cluster
                rowid = rowids[ind]
                #cluster_pts[curr_clust] = remove_array((obs, rowid), cluster_pts[curr_clust]) #removes obs from list of observations assigned to cluster
                cluster_pts[curr_clust] = remove_obs(rowid, cluster_pts[curr_clust]) #removes obs from list of observations assigned to cluster point
                #should return list of length -1
                cluster_pts[nearest_ind].append((obs, rowid, ind))  #adds to list of observations at new cluster point
        i += 1
    
    result = assigned_format(new_clusts, cluster_pts, k)
    #should be a list of the form [(cluster1, [pt1, pt2,..], [id1, id2, ...]), ...]
    return result 


def assigned_format(clusts, clust_pts, k):
    foo = [[0,[],[]] for _ in range(k)]
    for index, clust in enumerate(clusts): #clust is index of cluster
        foo[index][0] = clust
        for clust_pt in clust_pts[index]:
            foo[index][1].append(clust_pt[0])
            foo[index][2].append(clust_pt[1])
    return foo

"""

"""
def main():
    #python3 kmeans. <Filename> <K> [-p] [-m] [-n] [-t]

    rand = True
    euclid = True
    norm = False
    test = False

    if "-p" in sys.argv:
        rand = False
    if "-m" in sys.argv:
        euclid = False
    if "-n" in sys.argv:
        norm = True
    if "-t" in sys.argv:
        test = True
    
    path = sys.argv[1]
    k = int(sys.argv[2])
    df, groundtruth = parser(path)
    print("Ground Truth Values:", groundtruth.value_counts())
    matrix = encode_df(df, norm = norm)

    result = kmeans(matrix, k, rand, euclid, rowids = groundtruth) #[(cluster1, [pt1, pt2,..], [i1, i2, ...]), ...]

    mets = metrics(result, euclid)
    formatted = format_met(mets)
    if test:
        print(test_metrics(mets))
    else:
        output_mets(formatted)


if __name__ == "__main__":
    main()
