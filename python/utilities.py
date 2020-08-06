# Copyright 2019 by Erika Gardini.
# All rights reserved.
# This file is part of the Walking Through Music And Visual Art Spaces Tool,
# and is released under the "Apache License 2.0". Please see the LICENSE
# file that should have been included as part of this package.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import csv
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

DIR = "../datasets/"

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

music_dic = {1: "classical",
              2: "baroque",
              3: "rock",
              4: "opera",
              5: "medieval",
              6: "jazz"}

images_dic = {1: 'Early_Renaissance',
                2: 'Naïve_Art_(Primitivism)',
                3: 'Expressionism',
                4: 'Magic_Realism',
                5: 'Northern_Renaissance',
                6: 'Rococo',
                7: 'Ukiyo-e',
                8: 'Art_Nouveau_(Modern)',
                9: 'Pop_Art',
                10: 'High_Renaissance',
                11: 'Minimalism',
                12: 'Mannerism_(Late_Renaissance)',
                13: 'Art_Informel',
                14: 'Neoclassicism',
                15: 'Color_Field_Painting',
                16: 'Symbolism',
                17: 'Realism',
                18: 'Romanticism',
                19: 'Surrealism',
                20: 'Cubism',
                21: 'Impressionism',
                22: 'Baroque',
                23: 'Abstract_Expressionism',
                24: 'Post-Impressionism',
                25: 'Abstract_Art'}

def getMouseSamples2D(samples, labels, n, classes, name):
    """
    Get n points from samples by manual mouse selection.

    Args:
        [ndarray float] samples: data matrix

        [ndarray int] labels: labels in index format

        [int] n: number of points to be selected

        [ndarray int] classes: classes to consider

    Returns:
        [ndarray int]  ids: indices of the samples selected
    """
    ids = np.ndarray(n, int)
    n_sel = [n]

    pca = PCA(n_components=50, random_state=5)
    principalComponents = pca.fit_transform(samples)
    X = TSNE(n_components=2, random_state=20, perplexity=50, learning_rate=300).fit_transform(principalComponents)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    X_labels_names = getNamesFromClassNumbers(classes, labels, name)

    i = 0
    for label in (np.unique(X_labels_names)):
        ix = np.where(X_labels_names == label)
        if label != "Centroid":
            ax.scatter(X[ix, 0], X[ix, 1], c=colors[i], label=label)
            i = i + 1

    ix = np.where(X_labels_names == "Centroid")
    ax.scatter(X[ix, 0], X[ix, 1], c="black", label="Centroid")
    ax.legend()
    ax.set_title('Select ' + repr(n) + ' points')

    def onclick(ev):
        if (n_sel[0] == 0):
            plt.close()
            return

        n_sel[0] = n_sel[0] - 1

        mouseX = np.asarray([ev.xdata, ev.ydata], float)
        id_sel = np.argmin(np.diag(np.matmul(X[:, 0:2], X[:, 0:2].T)) - 2 * np.matmul(mouseX, X[:, 0:2].T))

        ax.plot(X[id_sel, 0], X[id_sel, 1], 'ro')
        if (n_sel[0] == 0):
            plt.title('Click to quit or close the figure')

        plt.draw()

        ids[n_sel[0]] = id_sel;

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(fig)
    fig.canvas.mpl_disconnect(cid)

    return ids

def getNamesFromClassNumbers(classes, X_labels, name):
    """
    Get the labels in string format given the labels in index format
    Args:
        [ndarray int] classes: classes to consider

        [ndarray int] X_labels: labels in index format

        [string] name:
            'music': work in music context
            'images': work in art context

    Returns:
        [ndarray string] X_labels_string: labels in string format
    """
    X_labels_string = []
    for el in X_labels:
        if el in classes:
            if name == 'music':
                X_labels_string.append(music_dic.get(el))
            elif name == 'images':
                X_labels_string.append(images_dic.get(el))

        else:
            X_labels_string.append("Centroid")

    X_labels_string = np.array(X_labels_string)
    return X_labels_string

def dataVisualization2D(X_2D, X_labels, centroids_2D, models_2D, model_euclidean_2D, classes, name, dir_name):
    """
    Plot all the datapoints, the principal path and the trivial path
    Args:
        [ndarray float] X_2D: data matrix

        [ndarray int] X_labels: labels in index format

        [ndarray float] centroids_2D: centrids matrix

        [ndarray float] models_2D: principal path matrix

        [ndarray float] model_euclidean_2D: trivial path matrix

        [ndarray int] classes: classes to consider

        [string] name:
            'music': work in music context
            'images': work in art context
    """
    fig = plt.figure(figsize=(10,8))
    ax2 = fig.add_subplot(1, 1, 1)

    X_labels_names = getNamesFromClassNumbers(classes, X_labels, name)

    i = 0
    unique_labels = getUniqueLabels(classes, name)

    for label in unique_labels:
        ix = np.where(X_labels_names == label)
        ax2.scatter(X_2D[ix,0], X_2D[ix,1], c=colors[i], label=label)
        i = i + 1

    ax2.plot(centroids_2D[:,0],centroids_2D[:,1],'ks', label="Centroids")
    ax2.plot(models_2D[:,0],models_2D[:,1],'-b*', label="Principal Path")
    ax2.plot(model_euclidean_2D[:,0],model_euclidean_2D[:,1],'-m^', label="Trivial Path")
    ax2.set_title('2D Data Visualization')
    ax2.axis('equal')
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fancybox=True, shadow=True, ncol=4)

    plt.savefig(dir_name+"/paths.svg")


def subdivideArray2D(X_embedded, X_labels, x_shape, NC):
    """
    Divide values in different matrices
    Args:
        [ndarray float] X_embedded: data matrix

        [ndarray int] X_labels: labels in index format

        [int] x_shape: number of datapoints

        [int] NC: number of waypoints - equal for both the paths

    Returns:
        [ndarray float] X_2D: datapoints matrix

        [ndarray int] X_labels_datapoints: labels in index format

        [ndarray float] centroids_2D: centroids matrix

        [ndarray float] models_2D: principal path matrix

        [ndarray float] model_euclidean_2D: trivial path matrix
    """
    X_2D = []
    centroids_2D = []

    for i in range(0, x_shape):
        if (X_labels[i] == 0):
            centroids_2D.append(X_embedded[i, :])
        else:
            X_2D.append(X_embedded[i, :])

    X_2D = np.array(X_2D)
    centroids_2D = np.array(centroids_2D)
    models_2D = X_embedded[x_shape:x_shape + NC, :]
    model_euclidean_2D = X_embedded[x_shape + NC:X_embedded.shape[0], :]

    X_labels_datapoints = []
    for i in range(0, X_labels.shape[0]):
        if (X_labels[i] != 0):
            X_labels_datapoints.append(X_labels[i])

    X_labels_datapoints = np.array(X_labels_datapoints)

    return X_2D, X_labels_datapoints, centroids_2D, models_2D, model_euclidean_2D

def removeCentroids(X, X_labels, info):
    """
    Remove the centroids from a matrix
    Args:
        [ndarray float] X: datapoints + centroids matrix

        [ndarray int] X_labels: datapoints labels + centroids labels in index format

        [ndarray string] info: datapoints info + centroids info matrix

    Returns:
        [ndarray float] X_red: datapoints matrix

        [ndarray int] X_labels_red: datapoints labels in index format

        [ndarray string] X_info_red: datapoints info matrix

    """
    X_red = []
    X_labels_red = []
    X_info_red = []
    for i in range(0, X.shape[0]):
        if X_labels[i] != 0:
            X_red.append(X[i, :])
            X_labels_red.append(X_labels[i])
            X_info_red.append(info[i, :])
    X_red = np.array(X_red)
    X_labels_red = np.array(X_labels_red)
    X_info_red = np.array(X_info_red)

    return X_red, X_labels_red, X_info_red

def findNeighbourhood(X, X_labels, model, info, classes, type, name, dir_name):
    """
     Find the neighbour for each waypoint
     Args:
         [ndarray float] X: datapoints + centroids matrix

         [ndarray int] X_labels: datapoints labels + centroids labels in index format

         [ndarray float] model: path matrix

         [ndarray string] info: datapoints info + centroids info matrix

         [ndarray int] classes: classes to consider

         [string] type:
            'pp': principal path
            'tp': trivial path

         [string] name:
            'music': work in music context
            'images': work in art context
     """
    [X_red, X_label_red, X_info_red] = removeCentroids(X, X_labels, info)

    knn = KNeighborsClassifier(n_neighbors=1, weights='distance').fit(X_red, X_label_red)
    res = knn.kneighbors(model, return_distance = False)

    #SAVE INFO
    file = open(dir_name + "/" + type + "_info.txt", "w")
    for i in range(0, res.shape[0]):
        file.write(str(X_info_red[res[i]]) + "\n")
    file.close()

    model_labels = knn.predict(model)
    plotNeighbourhood(classes, model_labels, type, name, dir_name)

def plotNeighbourhood(classes, model_labels, type, name, dir_name):
    """
     Plot the label assigned to each waypoint
     Args:
         [ndarray int] classes: classes to consider

         [ndarray int] model_labels: label of the waypoints

         [string] type:
            'pp': principal path
            'tp': trivial path

         [string] name:
            'music': work in music context
            'images': work in art context
     """
    print("Plotting KNN results")
    X_labels_names = getNamesFromClassNumbers(classes, model_labels, name)
    x_values = []
    for i in range(0, X_labels_names.shape[0]):
        x_values.append(i)
    x_values = np.array(x_values)

    fig = plt.figure(figsize=(8,3))
    ax3 = fig.add_subplot(1, 1, 1)
    ax3.set_ylim([0, 6])
    i = 1

    unique_labels = getUniqueLabels(classes, name)

    for label in unique_labels:
        ix = np.where(X_labels_names == label)

        y_values = model_labels[ix]
        for j in range(y_values.shape[0]):
            y_values[j] = i

        ax3.scatter(x_values[ix], y_values, c=colors[i-1], label=label)

        i = i + 1

    ax3.set_title('Recovered styles progression')
    ax3.legend()
    #plt.show()
    plt.savefig(dir_name + "/Recovered styles progression "+ type + ".svg")

def getUniqueLabels(classes, name):
    res = []
    for el in classes:
        if(name == 'music'):
            res.append(music_dic.get(el))
        elif(name == 'images'):
            res.append(images_dic.get(el))
    return np.array(res)

def getDataFromCsv(filename):
     """
     Open the csv and get the content in as array
     Args:
         [string] filename: name of the csv file
     Returns:
        [ndarray float] data: data matrix
     """

     with open(filename, newline='') as csvfile:
        img_style_art = csv.reader(csvfile)
        data = []
        for row in img_style_art:
            data.append(row)

     return np.array(data)

def getCentroidFromClassNumber(class_number, array, labels):
    """
    Get the centroid from a matrix
    Args:
        [int] class_number: index of the class
        [ndarray float] array: data matrix
        [ndarray int] labels: labels in index format
    Returns:
       [ndarray float] coor: centroid coordinates
    """

    subset_features = array[np.where(labels == class_number)]
    kmeans = KMeans(n_clusters=1, random_state=0).fit(subset_features)
    coor = kmeans.cluster_centers_;

    return coor

def getSubsetData(classes, features, labels, info, dates = None):
    """
    Get the subset of data according to the input classes
    Args:
        [int] class_number: index of the class
        [ndarray float] features: data matrix
        [ndarray int] labels: labels in index format
        [ndarray string] info: info matrix
        [ndarray int] dates: dates matrix

    Returns:
        [ndarray string] subset_info: info matrix
        [ndarray int] subset_labels: labels in index format
        [ndarray float] subset_features: data matrix
        [ndarray int] subset_dates: dates matrix
    """

    subset_labels = []
    subset_info = []
    subset_features = []
    subset_dates = []

    for i in range(0, features.shape[0]):
        if labels[i] in classes:
            subset_labels.append(labels[i])
            subset_features.append(features[i])
            subset_info.append(info[i, :])
            if dates is not None:
                subset_dates.append(dates[i])

    subset_info = np.array(subset_info)
    subset_labels = np.array(subset_labels)
    subset_features = np.array(subset_features).astype(np.float)
    if dates is not None:
        subset_dates = np.array(subset_dates).astype(np.int)

    return subset_info, subset_labels, subset_features, subset_dates


def getData(classes, mode, name):
    """
    Arrange the input for the algorithm
    Args:
         [ndarray int] classes: classes to consider

         [int] mode:
            0: from centroid to centroid
            1: chosen by the user
            2: from min date to max date (only for images)

         [string] name:
            'music': work in music context
            'images': work in art context
    Returns:
        [ndarray float] features: data matrix (with centroids)
        [ndarray int] labels: labels in index format
        [ndarray int] boundary_ids: indexes of the starting and the ending points
        [ndarray string] info: info for each datapoint
    """
    if mode == 2 and name == 'music':
        raise Exception("Mode 2 is available only for images")

    #Open datasets
    if name == 'music':
        array_data = getDataFromCsv(DIR + "music_dataset.csv")
        info = array_data[:, 0:2]
        features = array_data[:, 2:array_data.shape[1]].astype('float')
        labels = array_data[:,0].astype('int')
        [info, labels, features, dates] = getSubsetData(classes, features, labels, info)

    elif name == 'images':
        files = os.listdir(DIR)
        if("images_dataset.csv" not in files):
            fout = open(DIR + "images_dataset.csv", "a")
            for num in range(1, 10):
                f = open(DIR + "images_dataset_" + str(num) + ".csv")
                for line in f:
                    fout.write(line)
                f.close()
            fout.close()

        array_data = getDataFromCsv(DIR + "images_dataset.csv")

        info = array_data[:, 0:3]
        features = array_data[:, 3:array_data.shape[1]].astype('float')
        labels = array_data[:, 0].astype('int')
        dates = array_data[:, 2].astype('int')

        [info, labels, features, dates] = getSubsetData(classes, features, labels, info, dates)

    #Append centroids
    for el in classes:
        coor = getCentroidFromClassNumber(el, features, labels)
        features = np.concatenate((features, coor))
        labels = np.concatenate((labels, [0]))
        if name == 'music':
            info = np.concatenate((info, [['0', '-']]))
        elif name == 'images':
            info = np.concatenate((info, [['0', '-', '-']]))
            dates = np.concatenate((dates, [-1]))

    #Select boundaries
    boundary_ids = np.ndarray(2, int)
    if mode == 0:
        #Centroids
        boundary_ids[0] = features.shape[0] - classes.shape[0]
        boundary_ids[1] = features.shape[0] - 1
    elif mode == 1:
        #Visual selection
        boundary_ids=getMouseSamples2D(features, labels,2, classes, name)
    elif mode == 2: #only for images
        [index_min, index_max] = find_min_max_indexes(dates)
        #Select min and max dates as start and end points
        boundary_ids[0] = index_min
        boundary_ids[1] = index_max

    return features, labels, boundary_ids, info

def find_min_max_indexes(dates):
    """
    Get the index of the min value and the max value (centroids are not considered)
    Args:
        [ndarray int] dates: dates matrix

    Returns:
        [float] index_min: index of the min value
        [int] index_max: index of the max value
    """
    min = dates[0]
    index_min = 0
    max = dates[0]
    index_max = 0

    for i in range(0, dates.shape[0]):
        if dates[i] < min and dates[i] != -1:
            min = dates[i]
            index_min = i
        if dates[i] > max and dates[i] != -1:
            max = dates[i]
            index_max = i

    return index_min, index_max
