import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import spatial


ego414 = np.genfromtxt('414.featnames',dtype=str,delimiter="\n")
egofeature = []

file = open("414.featurenames.txt",'w')
for i in range(0,len(ego414)):
    egofeature.extend(ego414[i].split(';anonymized feature ')[0:-1])
    if (egofeature[i][-3:] == ';id'):
        egofeature[i] = egofeature[i].strip(';id')
    egofeature[i] = egofeature[i].replace(';','_')
    line = str(egofeature[i]) + '\n'
    file.write(line)
file.close()

ego_414 = np.genfromtxt('414.featurenames.txt',dtype=str,delimiter ="\n")
featnames = []
for j in range(0,len(ego_414)):
    featnames.append((ego_414[j].split())[1])

file = open("414featnames.txt",'w')
for i in range(0,len(featnames)):
    if i == len(featnames)-1:
        line = str(featnames[i])
    else:
        line = str(featnames[i])+'\n'
    file.write(line)
file.close()

### make feat_name list : feat_name

file = open("414featnames.txt",'r')
feat_name = file.read()
feat_name = feat_name.split('\n')
feat_name.insert(0,'nodes')
file.close()

### node no. , node attribute to array : features

features = []
feature = np.genfromtxt('414.feat',dtype = str, delimiter = "\n")

features = np.zeros((len(feature),len(feat_name)))
for i in range(0,len(feature)):
    features[i] = feature[i].split(' ')
features = features.astype('int32')

### read 414.edge : edges

edge = np.genfromtxt('414.edges',dtype = str, delimiter = "\n")
edges = np.zeros((len(edge),2))
for j in range(0,len(edge)):
    edges[j] = edge[j].split(' ')
edges = edges.astype('int32')
