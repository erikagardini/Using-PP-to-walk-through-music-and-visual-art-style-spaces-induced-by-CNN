# Copyright 2019 by Erika Gardini.
# All rights reserved.
# This file is part of the Walking Through Music And Visual Art Spaces Tool,
# and is released under the "Apache License 2.0". Please see the LICENSE
# file that should have been included as part of this package.

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from python import linear_utilities as lu, principalpath as pp
import utilities as pu
import sys
import os

def main(name, mode, classes, dir_name):
    print("Loading data...")
    [X, X_labels, boundary_ids, info] = pu.getData(classes, mode, name)

    NC = 100
    d = X.shape[1]

    ###PRINCIPAL PATH###

    print("Boundaries selection")
    #initialize waypoints
    waypoint_ids = lu.initMedoids(X, NC, 'kpp',boundary_ids)
    waypoint_ids = np.hstack([boundary_ids[0],waypoint_ids,boundary_ids[1]])
    W_init = X[waypoint_ids,:]

    print("Computing Principal Path")
    #annealing with rkm
    s_span=np.logspace(5,-5)
    s_span=np.hstack([s_span,0])
    models=np.ndarray([s_span.size,NC+2,d])

    for h,s in enumerate(s_span):
        [W,u]=pp.rkm(X, W_init, s, plot_ax=None)
        W_init = W
        models[h,:,:] = W
        print("Model " + str(h+1) + " out of " + str(len(s_span)))

    print("Performing model selection")
    #Variance
    W_dst_var = pp.rkm_MS_pathvar(models, s_span, X) #, W_dst_array
    s_elb_id = lu.find_elbow(np.stack([s_span, W_dst_var], -1))
    print("Model selected: " + str(s_elb_id))
    ###TRIVIAL PATH###

    #Simple path with euclidean distance
    print("Computing trivial path")
    model_euclidean = np.zeros([NC+2, d])
    lam = 0
    delta = 1/(NC+1)
    for h in range(0, NC+2):
        model_euclidean[h] = (lam * X[boundary_ids[1]]) + ((1-lam) * X[boundary_ids[0]])
        lam = lam + delta

    ###ASSIGN LABELS TO EACH PATH###
    print("Computing KNN for Principal Path")
    #KNN for model labeling
    pu.findNeighbourhood(X, X_labels, models[s_elb_id,:,:], info, classes, "pp", name, dir_name)

    print("Computing KNN for Trivial Path")
    pu.findNeighbourhood(X, X_labels, model_euclidean, info, classes, "tp", name, dir_name)

    ###2D DATA VISUALIZATION###
    print("Computing t-SNE")
    #Create the array to apply tSNE
    final_info = X
    final_info = np.concatenate((final_info, models[s_elb_id, 1:models.shape[1]-1, :]))
    final_info = np.concatenate((final_info, model_euclidean[1:model_euclidean.shape[0]-1,:]))

    #PCA reduction
    pca = PCA(n_components=50, random_state=5)
    principalComponents = pca.fit_transform(final_info)
    #tSNE reduction
    X_embedded = TSNE(n_components=2, random_state=20, perplexity=50, learning_rate=300).fit_transform(principalComponents)

    #Subdivide in X_2D, X_g_2D, centroids, models_2D and model with euclidean distance
    X_2D, X_labels_2D, centroids_2D, models_2D, model_euclidean_2D = pu.subdivideArray2D(X_embedded, X_labels, X.shape[0], NC)

    print("Plotting paths")
    #Plot path
    pu.dataVisualization2D(X_2D, X_labels_2D, centroids_2D, models_2D, model_euclidean_2D, classes, name, dir_name)

    print("FINE")

if __name__ == '__main__':
    name = sys.argv[1]
    mode = sys.argv[2]
    mode = int(mode)
    classes = []
    classes_names = ""
    for i in range(3, len(sys.argv)):
        classes.append(sys.argv[i])
        classes_names += str(sys.argv[i]) + "-"
    classes = np.array(classes).astype(np.int)
    classes_names = classes_names[:-1]
    dir_name = "../results/"+name+"/mode=" + str(mode) + "_" + classes_names
    if not os.path.isdir(dir_name):
        os.chdir("../results/"+name)
        os.mkdir("mode=" + str(mode) + "_" + classes_names)
    main(name, mode, classes, dir_name)
