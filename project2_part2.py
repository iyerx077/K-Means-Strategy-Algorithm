from precode2 import *
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import random

data = np.load('AllSamples.npy')
initial_centers = {}

for k in range(2, 11):
    centers = initial_S2("1670", k)  # please replace 0111 with your last four digit of your ID
    initial_centers[k] = centers


# calculate list of distances from centroids to datapoints; remove from datapoints once done
# def maxDistance()
def kMeans2(k1, initialcenters, sample):
    # setting up the first centroid in each cluster
    newcentroid = list(initialcenters[k1])
    sample = sample.tolist()
    # removing the first centroid to prevent duplication
    sample.remove(newcentroid)
    # finding the second centroid, which is the largest distance from first centroid to all datapoints
    dlist = [numpy.linalg.norm(sample[i] - initialcenters[k1]) for i in range(len(sample))]
    ind = numpy.argmax(dlist)
    newcentroid = [newcentroid, sample[ind]]  # to append to whole array and not just one element
    sample.remove(sample[ind])
    sample = numpy.array(sample)
    newcentroid = numpy.array(newcentroid)
    # since k=2 cluster has already been created, algorithm starts from k=3
    if k1 > 2:
        for k in range(k1 - 2):
            # list to collect the list of maximum averages
            maxavglist = []
            for j in range(len(sample)):
                # appending the mean of the distances
                maxavglist.append(maxmeandistance(newcentroid, sample[j]))
            # indexing the max mean distance
            ind = numpy.argmax(maxavglist)
            # concatenating old centroid with new one created
            newcentroid = numpy.concatenate((newcentroid, [sample[ind]]), axis=0)
            sample = sample.tolist()
            sample.remove(sample[ind])
            sample = numpy.array(sample)
            # centroid = numpy.array(centroid)
            # sample.remove(sample[ind])
    # finalcenters = [[] for i in range(k1)]
    # initialcenters[k1] = centroid
    return newcentroid


def maxmeandistance(array, datapoints):
    # list of all the distances from current centroid points
    distlist = [numpy.linalg.norm(array[i] - datapoints) for i in range(len(array))]
    return numpy.mean(distlist)


# code below is from part 1 kmeans algorithm
def kMeans(k1, initcenters, sample):  # number of clusters,center value,sample
    index = 0
    loss = 1
    while (loss > 0) and (index < 100):
        # initialize dictionary of 0s based off of cluster number
        cluster = genCluster(k1)
        for s in range(len(sample)):
            # classify indexed datapoint to corresponding cluster
            cluster[mindistindex(k1, sample, initcenters, s)].append(sample[s])
        # print (cluster)
        emptyarray = [[0, 0] for i in range(k1)]
        oldcenters = convertformat(k1, initcenters, emptyarray)

        # updating the input centers
        for i in range(k1):
            initcenters[k1][i][0] = numpy.mean(cluster[i], axis=0)[0]  # initcenters[k1][i][0]
            initcenters[k1][i][1] = numpy.mean(cluster[i], axis=0)[1]  # initcenters[k1][z][1]

        # calculating the loss function; will end the loop if it is 0, means centroids have done changing
        loss = lossFunction(k1, oldcenters, initcenters[k1])
        index += 1
        finalcenters = convertformat(k1, initcenters, emptyarray)
        finalcluster = [[] for i in range(k1)]
        # finalcenters for proper formatting for output
        for j in range(k1):
            for z in range(len(cluster[j])):
                finalcluster[j].append(cluster[j][z])
    # returning the final centers calculated and cluster for loss calculation
    return [finalcenters, finalcluster]


# differences must be 0 between each point
# go from each center and assign to cluster based on closest points to center; take mean of cluster and keep iterating until zero changes

# closest datapoints to the centers added to cluster; take mean of them

def convertformat(k, oldarray, newarray):
    for i in range(k):
        newarray[i][0] = oldarray[k][i][0]
        newarray[i][1] = oldarray[k][i][1]
    return newarray


def mu(cluster, sample, center):
    r = 0
    for s in range(len(sample)):
        if cluster == numpy.argmin(numpy.linalg.norm(center[0] - sample[s])):
            r += 1
    return r


def genCluster(k):
    cluster = {}
    for i in range(k):
        cluster[i] = []
    return cluster


def mindistindex(k, sample, centers, index):
    # list of minimum distances from each centroid to datapoints
    dlist = [numpy.linalg.norm(sample[index] - centers[k][j]) for j in range(k)]
    # find the index of the minimum distance of the centroid to the datapoint
    ind = numpy.argmin(dlist)
    return ind


def objectivefun(k, init, cluster):  # takes in cluster value, centroids, cluster array
    total = 0
    # objectivefun(2,10,initial_centers[10],kMeans(10,initial_centers,data))
    # iterates through the clusters and calculates loss compared to input centroids
    for i in range(k):
        for c in range(len(cluster[i])):
            #            print (init[i])
            #           print (cluster[i][c])
            total += numpy.linalg.norm(init[i] - cluster[i][c]) ** 2
    return total


def lossFunction(k, point, sets):
    outersum = 0
    innersum = 0
    # calculates loss to check when centroids stop updating
    for i in range(k):
        for p in range(len(sets)):
            innersum += numpy.linalg.norm(point[p] - sets[p])
        outersum += innersum
    return outersum


# print (kMeans2(10, initial_centers, data))
'''test_centers[2] = kMeans2(2, initial_centers, data)
test_centers[3] = kMeans2(3, initial_centers, data)
test_centers[4] = kMeans2(4, initial_centers, data)
test_centers[5] = kMeans2(5, initial_centers, data)
test_centers[6] = kMeans2(6, initial_centers, data)
test_centers[7] = kMeans2(7, initial_centers, data)
test_centers[8] = kMeans2(8, initial_centers, data)
test_centers[9] = kMeans2(9, initial_centers, data)
test_centers[10] = kMeans2(10, initial_centers, data)'''
test_centers = {}
for i in range(2, 11):
    test_centers[i] = kMeans2(i, initial_centers, data)

final_centersstrat2 = {}
final_clustersstrat2 = {}
for k in range(2,11):
    final_centersstrat2[k] = kMeans(k,test_centers,data)[0]
    final_clustersstrat2[k] = kMeans(k,test_centers,data)[1]
    print (final_centersstrat2[k])

lossesstrat2 = {}
for k in range(2,11):
    lossesstrat2[k] = objectivefun(k,kMeans(k,test_centers,data)[0],kMeans(k,test_centers,data)[1])
    print (lossesstrat2[k])

for i in range(len(final_clustersstrat2[2])):
    for c in range(len(final_clustersstrat2[2][i])):
        plt.scatter(final_clustersstrat2[2][i][c][0], final_clustersstrat2[2][i][c][1], s=100, color='green')

for i in range(len(final_centersstrat2[2])):
    plt.scatter(final_centersstrat2[2][i][0], final_centersstrat2[2][i][1], s=60, color='black')

plt.ylim(-5, 15)
plt.xlim(-5, 15)
plt.title("Final Center vs Cluster for k=2 using K-Means Strategy 2")
plt.show()

for i in range(len(final_clustersstrat2[10])):
    for c in range(len(final_clustersstrat2[10][i])):
        plt.scatter(final_clustersstrat2[10][i][c][0], final_clustersstrat2[10][i][c][1], s=100, color='red')

for i in range(len(final_centersstrat2[10])):
    plt.scatter(final_centersstrat2[10][i][0], final_centersstrat2[10][i][1], s=60, color='black')

plt.ylim(-5, 15)
plt.xlim(-5, 15)
plt.title("Final Center vs Cluster for k=10 using K-Means Strategy 2")
plt.show()

for c in range(2, 11):
    plt.scatter(c, lossesstrat2[c], color='red')

plt.ylabel("Loss")
plt.xlabel("Clusters")
plt.title("Final Loss vs Clusters Using k-Means Strategy 2")
plt.show()