from functions import *


#python3 dbscan.py <Filename> <epsilon> <NumPoints>


def DBScan(D, dist_func, eps, minpts):
    #D = dataset, dist_func = how to calculate distances between points
    #eps = epsilon, the radius within which DBScan shall search for points
    #minpts = the min number of poibts within the epsilon distance from a given point to continue building the cluster 

    #Core point discovery --> scans throug the entire dataset and makes a list of all the core points
    core = {} #key = point index, value = all its neighboring points within episilon distance
    clusters = {} #key = point index, value = its cluster
    matrix = calc_dist_mat(D, dist_func) #D is a numpy matrix, 
    
    rownum = matrix.shape[0]
    #Finding core points and initializing cluster (0) for every point
    for row in range(rownum):
        #Check if core point
        clusters[row] = 0
        neighbor_indices = np.where(matrix[row] <= eps)[0]
        dict_values = neighbor_indices.tolist()    
        dict_values.remove(row)
        # Check if point is a core point
        if len(dict_values) >= minpts:
            dict_values = neighbor_indices.tolist()
            core[row] = dict_values

    ###2
    CurrentCluster = 0
    core_keys = list(core.keys())
    for d in core_keys: #d = index of point
        if clusters[d] == 0: 
            CurrentCluster += 1 #start a new cluster
            clusters[d] = CurrentCluster #assign first point to the cluster
            clusters = DensityConnected(d, core_keys, core, CurrentCluster, clusters) #find all density connected points
    
    ###3 
    clusterList = []
    for k in range(1, CurrentCluster+1):
        cluster_k = []
        for index, d in enumerate(matrix):
            if clusters[index] == k:
                cluster_k.append(index)
        if len(cluster_k) != 0:
            clusterList.append(cluster_k)
    

    #Noise:= {d ∈ D|cluster(d) =k}}
    noise = []
    for point, cluster in clusters.items(): 
        if cluster == 0: 
            noise.append(point)

    #Border:= D – (Noise U Core)
    border = [row for row in range(rownum) if row not in core.keys() and row not in noise]

    return clusterList, core, clusters, noise, border

def DensityConnected(point, core_keys, core, clusterID, clusters):
    if len(core_keys) == 0:
        return clusters
    
    for d in core[point]: #for d in list of neighbors to the point
        clusters[d] = clusterID 
        if d in core_keys:
            core_keys.remove(d) #removing the neighbor from the "queue"
            clusters = DensityConnected(d, core_keys, core, clusterID, clusters)
    return clusters


def main():
    euclid = True
    test = False

    path = sys.argv[1]
    eps = float(sys.argv[2])
    minpts = int(sys.argv[3])

    if "-t" in sys.argv:
        test = True

    df, ground_truth = parser_DBScan(path)

    matrix = encode_df(df, norm = False)

    clusterlist, core, clusters, noise, border = DBScan(matrix, euclid, eps, minpts)
    cluster_pts = []
    for cluster in clusterlist:
        cluster_pts.append([np.array(df.iloc[i]) for i in cluster])
    new_clusts = np.empty((len(clusterlist), matrix.shape[1]))
    for index, points in enumerate(cluster_pts): #points is list of numpy arrays
        if points: #not empty
            mean_values = np.mean(points, axis=0) 
            new_clusts[index] = mean_values
    result = list(zip(new_clusts, cluster_pts, clusterlist))
    mets = metrics(result, euclid)
    formatted = format_met(mets)
    if test:
        print(test_metrics(mets))
    else:
        output_mets(formatted)

    print("epsilon: ", eps, " and minpts: ", minpts)
    print("\n")

    no_all_points = len(core.keys()) + len(noise) + len(border)
    noise_points = (len(noise) / no_all_points)*100
    border_points = (len(border) / no_all_points)*100
    
    print("noise points: ", noise)
    print("no noise points: ", noise_points, "‰")
    print("\n")
    
    print("border points: ", border)
    print("no border points: ", border_points, "‰")
    print("\n")

    if ground_truth is not None:
        ground_truth_list = ground_truth.to_list()
        
        #Printing all the found clusters
        for j in range(len(clusterlist)):
            names = [ground_truth_list[index] for index in clusterlist[j]]
            print(f"cluster {j} :", clusterlist[j], " which corresponds to: \n", names)
            print("\n")
        
    else:
        #Printing all the found clusters
        for j in range(len(clusterlist)):
            print(f"cluster {j} :", clusterlist[j])
            print("\n")
        
    cluster_points = ((no_all_points-(len(noise) + len(border)))/no_all_points)*100
    print("number of clustered points: ", cluster_points, "‰")


if __name__ == "__main__":
    main()

