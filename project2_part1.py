# import code here
from precode import *
import numpy
import pandas as pd
import random
import matplotlib.pyplot as plt

data = np.load('AllSamples.npy')
initial_centers = {}
for k in range(2, 11):
    centers = initial_S1("1670", k)  # please replace 0111 with your last four digit of your ID
    initial_centers[k] = centers
def kMeans(k1,initcenters,sample):#number of clusters,center value,sample
    index = 0
    loss = 1
    while (loss >0) and (index<100):
    #initialize dictionary of 0s based off of cluster number
        cluster = genCluster(k1)
        for s in range(len(sample)):
            #classify indexed datapoint to corresponding cluster
            cluster[mindistindex(k1,sample,initcenters,s)].append(sample[s])
        #print (cluster)
        emptyarray = [[0,0] for i in range(k1)]
        oldcenters = convertformat(k1,initcenters,emptyarray)
        '''for i in range(k1):
            oldcenters[i][0]=initcenters[k1][i][0]
            oldcenters[i][1]=initcenters[k1][i][1]'''

        #updating the input centers
        for i in range(k1):
            initcenters[k1][i][0]=numpy.mean(cluster[i],axis=0)[0]#initcenters[k1][i][0]
            initcenters[k1][i][1]=numpy.mean(cluster[i],axis=0)[1]#initcenters[k1][z][1]

        #calculating the loss function; will end the loop if it is 0, means centroids have done changing
        loss = lossFunction(k1,oldcenters,initcenters[k1])
        index += 1
        finalcenters = convertformat(k1,initcenters,emptyarray)
        finalcluster = [[] for i in range (k1)]
        #finalcenters for proper formatting for output
        '''for i in range(k1):
            finalcenters[i][0]=initcenters[k1][i][0]
            finalcenters[i][1]=initcenters[k1][i][1]'''
        for j in range(k1):
            for z in range(len(cluster[j])):
                finalcluster[j].append(cluster[j][z])
    #returning the final centers calculated and cluster for loss calculation
    return [finalcenters,finalcluster]

#differences must be 0 between each point
#go from each center and assign to cluster based on closest points to center; take mean of cluster and keep iterating until zero changes

#closest datapoints to the centers added to cluster; take mean of them

def convertformat(k, oldarray, newarray):
    for i in range(k):
        newarray[i][0] = oldarray[k][i][0]
        newarray[i][1] = oldarray[k][i][1]
    return newarray

def mu(cluster,sample,center):
    r = 0
    for s in range(len(sample)):
        if cluster==numpy.argmin(numpy.linalg.norm(center[0]-sample[s])):
            r += 1
    return r
def genCluster(k):
    cluster = {}
    for i in range(k):
        cluster[i] = []
    return cluster

def mindistindex(k,sample,centers,index):
    #list of minimum distances from each centroid to datapoints
    dlist = [numpy.linalg.norm(sample[index]-centers[k][j]) for j in range(k)]
    #find the index of the minimum distance of the centroid to the datapoint
    ind = numpy.argmin(dlist)
    return ind
def objectivefun(k,init,cluster):#takes in cluster value, centroids, cluster array
    total = 0
    #objectivefun(2,10,initial_centers[10],kMeans(10,initial_centers,data))
    #iterates through the clusters and calculates loss compared to input centroids
    for i in range(k):
        for c in range(len(cluster[i])):
#            print (init[i])
 #           print (cluster[i][c])
            total += numpy.linalg.norm(init[i]-cluster[i][c])**2
    return total

def lossFunction(k,point,sets):
    outersum = 0
    innersum = 0
    #calculates loss to check when centroids stop updating
    for i in range(k):
        for p in range(len(sets)):
            innersum += numpy.linalg.norm(point[p]-sets[p])
        outersum += innersum
    return outersum
### TEST FUNCTION: test_question1
# DO NOT REMOVE THE ABOVE LINE
final_centers = {}
final_clusters = {}
for k in range(2,11):
    final_centers[k] = kMeans(k,initial_centers,data)[0]
    final_clusters[k] = kMeans(k,initial_centers,data)[1]
    print (kMeans(k,initial_centers,data)[0])
### TEST FUNCTION: test_question2
# DO NOT REMOVE THE ABOVE LINE
losses = {}
for k in range(2,11):
    losses[k] = objectivefun(k,kMeans(k,initial_centers,data)[0],kMeans(k,initial_centers,data)[1])
    print (objectivefun(k,kMeans(k,initial_centers,data)[0],kMeans(k,initial_centers,data)[1]))
for i in range(len(final_clusters[2])):
    for c in range(len(final_clusters[2][i])):
        plt.scatter(final_clusters[2][i][c][0], final_clusters[2][i][c][1], s=100, color='green')

for i in range(len(final_centers[2])):
    plt.scatter(final_centers[2][i][0], final_centers[2][i][1], s=60, color='black')

plt.ylim(-5, 15)
plt.xlim(-5, 15)
plt.title("Final Center vs Cluster for k=2 using K-Means Algorithm")
plt.show()
for i in range(len(final_clusters[10])):
    for c in range(len(final_clusters[10][i])):
        plt.scatter(final_clusters[10][i][c][0], final_clusters[10][i][c][1], s=100, color='red')

for i in range(len(final_centers[10])):
    plt.scatter(final_centers[10][i][0], final_centers[10][i][1], s=60, color='black')

plt.ylim(-5, 15)
plt.xlim(-5, 15)
plt.title("Final Center vs Cluster for k=10 using K-Means Algorithm")
plt.show()
for c in range(2, 11):
    plt.scatter(c, losses[c], color='red')

plt.ylabel("Loss")
plt.xlabel("Clusters")
plt.title("Final Loss vs Clusters")
plt.show()