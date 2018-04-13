import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import csv
from pandas import DataFrame
from scipy import spatial
from sklearn.preprocessing import scale


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

### make graph using NetworkX

G = nx.Graph()
nodes = features[:,0]
attributes = features[:,1:]

### make circle file AND find GROUND TRUTH alpha & beta
circle414 = np.genfromtxt('414.circles',dtype=str,delimiter="\n")
cluster = []
cluster_ele = []
cluster_total = []
outlier = np.zeros((1,len(nodes)))

file = open("414circle.txt",'w')
for i in range(0,len(circle414)):
    cluster.append(circle414[i].split('\t')[0][-1])
    for j in range(1,len(circle414[i].split('\t'))):
        cluster_ele.append(circle414[i].split('\t')[j])
        cluster_total.append(circle414[i].split('\t')[j])
    #file.write(cluster[i] + '\n')
        
    for k in range(0, len(cluster_ele)):
        if k == len(cluster_ele)-1:
            file.write(cluster_ele[k] + '\n')
        else:
            file.write(cluster_ele[k] + ' ')
        
    cluster_ele = []

file.close()

#- make ground truth cluster matrix
clustermatrix = np.zeros((159,7))
f = open('414circle.txt','r')
line = f.readline()
i = 0
while line:
    
    line = line.split(' ')
    line[len(line)-1] = line[len(line)-1][:-1]
    for j in range(0,len(line)):
        clustermatrix[np.where(nodes == int(line[j]))[0][0],i] = clustermatrix[np.where(nodes == int(line[j]))[0][0],i] + 1
    line = f.readline()
    i = i + 1
np.savetxt('original_cluster.csv',clustermatrix,fmt = '%d', delimiter = ",")
f.close()

#- ground truth cluster number
print('ground truth cluster number is ' + str(len(cluster)))

#- ground truth beta value
for i in range(0,len(cluster_total)):
    #- if outlier value is 0, it means that node is outlier
    outlier[0][np.where(nodes == int(cluster_total[i]))[0][0]] = outlier[0][np.where(nodes == int(cluster_total[i]))[0][0]] +1
beta = len(np.where(outlier == 0))/len(nodes)
print('ground truth beta value is ' + str(beta))

#- ground truth alpha value
alpha = len(cluster_total)/len(nodes) - 1
print('ground truth alpha value is ' + str(alpha))

#- ADD node and node's attributes
for i in range(0,len(nodes)):
    G.add_node(nodes[i],attr = attributes[i])

#- Add edge
G.add_edges_from(edges)

# make an adjacency matrix about original graph
original_adj = np.zeros((len(nodes),len(nodes)))

for (i,j) in G.edges:
    original_adj[np.where(nodes == i)[0][0],np.where(nodes == j)[0][0]] = original_adj[np.where(nodes == i)[0][0],np.where(nodes == j)[0][0]] + 1
    original_adj[np.where(nodes == j)[0][0],np.where(nodes == i)[0][0]] = original_adj[np.where(nodes == j)[0][0],np.where(nodes == i)[0][0]] + 1

np.savetxt("original_adj.csv",original_adj, fmt = '%d',delimiter = ",")

### Find what detailed features that have more than 90% jaccard similarity
print('original feature length is :', len(features[0])-1)

simindex = np.zeros((2,1))
for i in range(1,len(features[0])-1):
    for j in range(i+1,len(features[0])):
        
        M01 = 0
        M10 = 0
        M11 = 0
        for k in range(0,len(features[:,i])):
            if features[:,i][k] == 0 and features[:,j][k] == 1:
                M01 = M01 + 1
            elif features[:,i][k] == 1 and features[:,j][k] == 0:
                M10 = M10 + 1
            elif features[:,i][k] == 1 and features[:,j][k] == 1:
                M11 = M11 + 1
        
        sim = M11 / (M01 + M10 + M11)
        if sim > 0.9:
            print(feat_name[i] + " is similar with " + feat_name[j] + " and feature indexs are " +  str(i),str(j)
                  + " and number of features is " + str(feat_name.count(feat_name[i])), str(feat_name.count(feat_name[j])))
            
            simindex = np.c_[simindex,np.array([i,j])]

featurelist = ['birthday','education_concentration','education_degree','education_school',
               'education_type','education_year','first_name','gender','hometown','languages',
               'last_name','locale','location','name','work_employer','work_end_date','work_location',
               'work_position','work_start_date']

#- Then pop one column since these two columns are redundant
Suff = 0

for i in range(1,len(simindex[1,:])):
    
    if feat_name.count(feat_name[int(simindex[1,i])]) == 1:
        index = featurelist.index(feat_name[int(simindex[1,i])])
        featurelist.pop(index)
        
    feat_name.pop(int(simindex[1,i]))
    
    Suff = Suff + 1
    for j in range(i+1,len(simindex[1,:])):
        simindex[1,j] = simindex[1,j] - Suff
    
    features = np.c_[features[:,:int(simindex[1,i])],features[:,int(simindex[1,i])+1:]]

### calculate jaccard similarity if there's an edge

#- note : feature names are (19)
#-      : birthday / education_concentration / education_degree / education_school / education_type / education_year
#-      : first_name / gender / hometown / languages / last_name / locale / location / name / work_employer
#-      : work_end_date / work_location / work_position / work_start_date

edges = list(G.edges)

similarity = np.zeros((len(edges),1+len(features[0])))
nodesC2 = int(len(nodes)*(len(nodes)-1)/2)
whole_similarity = np.zeros((nodesC2,1+len(features[0])))

#######################################  NEW Feature Weight MEASURE ############################################################
onlyEdge = np.zeros((1,len(features[0])-1))
whole = np.zeros((1,len(features[0])-1))

for n1_index in range(0,len(nodes)):
    for n2_index in range(n1_index+1,len(nodes)):

        for k in range(1,len(feat_name)):
            if features[n1_index][k] == 1 and features[n2_index][k] == 1:
                whole[0][k-1] = whole[0][k-1] + 1

for i in range(0,len(edges)):

    N1 = edges[i][0]
    N2 = edges[i][1]
    n1_index = int(np.where(nodes == N1)[0])
    n2_index = int(np.where(nodes == N2)[0])

    for k in range(1,len(feat_name)):
        if features[n1_index][k] == 1 and features[n2_index][k] == 1:
            onlyEdge[0][k-1] = onlyEdge[0][k-1] + 1

ratio_whole = np.zeros((1,len(features[0])-1))
ratio_onlyEdge = np.zeros((1,len(features[0])-1))

sum_whole = sum(whole[0])
sum_onlyEdge = sum(onlyEdge[0])

for i in range(0,len(whole[0])):
    ratio_whole[0][i] = ratio_whole[0][i] + round((whole[0][i]/sum_whole)*100,5)
for j in range(0,len(onlyEdge[0])):
    ratio_onlyEdge[0][j] = ratio_onlyEdge[0][j] + round((onlyEdge[0][j]/sum_onlyEdge)*100,5)

upperindex = []
uppervalue = []

for k in range(0,len(whole[0])):
    if ratio_onlyEdge[0][k] - ratio_whole[0][k] > 0:
        upperindex.append(k)
        uppervalue.append(ratio_onlyEdge[0][k]- ratio_whole[0][k])

##### write to csv file

upperfile = open('UpperIndexValue.csv','w',newline='')
wr = csv.writer(upperfile)
for i in range(0,len(upperindex)):
    wr.writerow([upperindex[i],uppervalue[i]])
upperfile.close()

## take upper 25% value of uppervalue because upper 25% occupy 93% of influences
#- upper 25%
uppervalue_mean = np.mean(uppervalue)
uppervalue_std = np.std(uppervalue)

uppervalue_gaussian = scale(uppervalue)
uppervalue_gaussian_mean = np.mean(uppervalue_gaussian)
uppervalue_gaussian_std = np.std(uppervalue_gaussian)

uppervalue_upper25_index = []
uppervalue_upper25_value = sorted(uppervalue, reverse = True) ## sorting descending order
uppervalue_upper25_value = uppervalue_upper25_value[0:round(0.25*len(uppervalue))] ## because of upper 25%
for i in range(0,len(uppervalue_upper25_value)):
    #################################################################################### Corrected
    uppervalue_upper25_index.append(upperindex[np.where(uppervalue_upper25_value[i] == uppervalue)[0][0]])

uppervalue_norm = []
for i in range(0,len(uppervalue_upper25_index)):
    uppervalue_norm.append(uppervalue_upper25_value[i]/sum(uppervalue_upper25_value))

upper = np.zeros((1,len(feat_name)-1))
for i in range(0,len(uppervalue_upper25_index)):
    upper[0][uppervalue_upper25_index[i]] = upper[0][uppervalue_upper25_index[i]] + 1

k = 0
for n1_index in range(0,len(nodes)):
    for n2_index in range(n1_index+1,len(nodes)):
        n1 = nodes[n1_index]
        n2 = nodes[n2_index]
        whole_similarity[k,0] = whole_similarity[k,0] + n1
        whole_similarity[k,1] = whole_similarity[k,1] + n2

        end = 1

        for n in range(0,len(uppervalue_upper25_index)):

            if features[n1_index, uppervalue_upper25_index[n] + 1] == 1 and features[n2_index,uppervalue_upper25_index[n] + 1] == 1:
                whole_similarity[k][uppervalue_upper25_index[n]+2] = whole_similarity[k][uppervalue_upper25_index[n]+2] + uppervalue_norm[n]
        k = k + 1

for i in range(0,len(edges)):
    
    N1 = edges[i][0]
    N2 = edges[i][1]
    similarity[i][0] = N1
    similarity[i][1] = N2
    N1_index = int(np.where(nodes == N1)[0])
    N2_index = int(np.where(nodes == N2)[0])

    end = 1

    for n in range(0,len(uppervalue_upper25_index)):
        
        if features[N1_index,uppervalue_upper25_index[n] + 1] == 1 and features[N2_index,uppervalue_upper25_index[n] + 1] == 1:
            similarity[i][uppervalue_upper25_index[n] + 2] = similarity[i][uppervalue_upper25_index[n] + 2] + uppervalue_norm[n]
                   

############################# NEW WEIGHT MEASURE ##########################
for i in range(0,len(edges)):
    
    G.add_edge(similarity[i][0],similarity[i][1], weight = 1) ### 1: Edge  similarity[i][2:]

###########################################################################


### if there is NO edge between node pairs, then calcuate "edge weight" value
### if edge weight value is more higher, then make an edge and add edge weight
NoEdgelist_index = []
simlist = []

for i in range(0,len(similarity)):
    simlist.append(list(similarity[i,0:2]))

for j in range(0,len(whole_similarity)):
    if list(whole_similarity[j,0:2]) not in simlist:
        NoEdgelist_index.append(j)


NoEdgeSimilarity = np.zeros((len(NoEdgelist_index),len(whole_similarity[0])))

# Make array "NoEdgeSimilarity" which is similarity of node pair that has no edge
for j in range(0,len(NoEdgelist_index)):
    NoEdgeSimilarity[j,:] = whole_similarity[NoEdgelist_index[j],:]

# Assume there exist all edges for NO edge node pair,
# then calculate edge weight
NoEdgeWeight = np.zeros((len(NoEdgelist_index),3))

####################### NEW ###################################

for k in range(0,len(NoEdgelist_index)):
    NoEdgeWeight[k,0] = NoEdgeSimilarity[k,0]
    NoEdgeWeight[k,1] = NoEdgeSimilarity[k,1]    
    NoEdgeWeight[k,2] = sum(NoEdgeSimilarity[k][2:])

################################################################


# modify NoEdgeWeight value to Gaussian distribution
# upper 32%(upper average) , add edge and add edge weight to graph

NoEdgeWeight_mean = np.mean(NoEdgeWeight[:,2])
NoEdgeWeight_std = np.std(NoEdgeWeight[:,2])

NoEdgeWeight_gaussian = scale(NoEdgeWeight[:,2])
NoEdgeWeight_gaussian_mean = np.mean(NoEdgeWeight_gaussian)
NoEdgeWeight_gaussian_std = np.std(NoEdgeWeight_gaussian)

added_index = []

NWE = sorted(NoEdgeWeight[:,2],reverse=True)
######################## PLOT ############################
NWE_xaxis = []
NWE_ascendingorder = sorted(NoEdgeWeight[:,2])
NWE_ascendingsum = []
threshold = 0.06
threshold_index = []
for i in range(0,len(NWE_ascendingorder)):
    NWE_xaxis.append(i)
##    NWE_ascendingsum.append(sum(NWE_ascendingorder))
##    if i != 0 and NWE_ascendingorder[i] - NWE_ascendingorder[i-1] > threshold :
##        threshold_index.append(i)
plt.scatter(NWE_xaxis,NWE_ascendingorder, s = 5, marker = 'o')
plt.xlabel('index')
plt.ylabel('NoEdgeNodePairSimilarity')
plt.show()

########################  NEW  ############################

NWE_truncated = NWE[0:round(0.32*len(NWE))] ## because of upper 32%

############################################################
NWE_index = []
for i in range(0,len(NWE_truncated)):
    NWE_index.append(np.where(NWE_truncated[i] == NoEdgeWeight[:,2])[0][0])


for i in range(0,len(NWE_index)):
        
    G.add_edge(NoEdgeWeight[NWE_index[i],0],NoEdgeWeight[NWE_index[i],1], weight = 1)
    added_index.append(i)

### make adjacency matrix
adjacency_matrix = nx.adjacency_matrix(G, nodelist = nodes, weight = 'weight')
adjacency_matrix = adjacency_matrix.todense()
adj_dataframe = pd.DataFrame(adjacency_matrix, index = nodes, columns = nodes)
adj_dataframe.to_csv('ego414_adjacencymatrix.csv')

### make adjacency matrix about added edge, weigthed graph

added_edge_adj = np.zeros((len(nodes),len(nodes)))

for (i,j) in G.edges:
    added_edge_adj[np.where(nodes == i)[0][0],np.where(nodes == j)[0][0]] = added_edge_adj[np.where(nodes == i)[0][0],np.where(nodes == j)[0][0]] + 1 #+ round(G.edges[i,j]['weight'],2)*100
    added_edge_adj[np.where(nodes == j)[0][0],np.where(nodes == i)[0][0]] = added_edge_adj[np.where(nodes == j)[0][0],np.where(nodes == i)[0][0]] + 1 #+round(G.edges[i,j]['weight'],2)*100

# adjacency matrix to csv file
#np.savetxt("v5_weighted_adj.csv",weighted_adj,fmt = '%d',delimiter = ",")
np.savetxt("NNEW_32%edge_added_adj.csv",added_edge_adj,fmt = '%d' ,delimiter = ",")

### plot Graph

added = np.zeros((len(added_index),3))
for j in range(0,len(added_index)):
    added[j,0] = NoEdgeWeight[added_index[j],0]
    added[j,1] = NoEdgeWeight[added_index[j],1]
    added[j,2] = NoEdgeWeight[added_index[j],2]

added_large = []
added_small = []
for k in range(0,len(added)):
    if added[k,2] > np.mean(added[:,2]):
        added_large.append((added[k,0],added[k,1]))
    else:
        added_small.append((added[k,0],added[k,1]))
			
pos = nx.spring_layout(G,dim=2)

nx.draw_networkx_nodes(G, pos, node_size = 1)

### elarge : black edge
##nx.draw_networkx_edges(G, pos, edgelist = elarge, width = 1,alpha = 1)
### esmall : blue dotted edge
##nx.draw_networkx_edges(G, pos, edgelist = esmall, width = 1, alpha = 0.4, edge_color='b', style = 'dotted')
nx.draw_networkx_edges(G, pos, edgelist = edges, width = 1, alpha = 1)
# added_large : green dotted edge
nx.draw_networkx_edges(G, pos, edgelist = added_large, width = 1, alpha = 0.3, edge_color = 'b', style = 'dotted')
# added_small : yellow dteed edge
nx.draw_networkx_edges(G, pos, edgelist = added_small, width = 1, alpha = 0.3, edge_color = 'b', style = 'dotted')

#nx.draw_neworkx_labels(G, pos, font_size = 10, font_family = 'sans-serif')

plt.axis('off')
plt.show()
