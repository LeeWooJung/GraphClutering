import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
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

#- ADD node and node's attributes
for i in range(0,len(nodes)):
    G.add_node(nodes[i],attr = attributes[i])

#- Add edge
G.add_edges_from(edges)

### Find what detailed features that have more than 90% jaccard similarity
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
for i in range(1,len(simindex[1,:])):
    
    if feat_name.count(feat_name[int(simindex[1,i])]) == 1:
        index = featurelist.index(feat_name[int(simindex[1,i])])
        featurelist.pop(index)
    feat_name.pop(int(simindex[1,i]))
    features = np.c_[features[:,:int(simindex[1,i])],features[:,int(simindex[1,i])+1:]]

### calculate jaccard similarity if there's an edge

#- note : feature names are (19)
#-      : birthday / education_concentration / education_degree / education_school / education_type / education_year
#-      : first_name / gender / hometown / languages / last_name / locale / location / name / work_employer
#-      : work_end_date / work_location / work_position / work_start_date

edges = list(G.edges)

similarity = np.zeros((len(edges),2+19))
nodesC2 = int(len(nodes)*(len(nodes)-1)/2)
whole_similarity = np.zeros((nodesC2,2+19))

k = 0
for n1_index in range(0,len(nodes)):
    for n2_index in range(n1_index+1,len(nodes)):
        n1 = nodes[n1_index]
        n2 = nodes[n2_index]
        whole_similarity[k,0] = whole_similarity[k,0] + n1
        whole_similarity[k,1] = whole_similarity[k,1] + n2

        end = 1

        for n in range(0,len(featurelist)):
            count = 0
            start = end
            num_of_feature = feat_name.count(featurelist[n])
            end = start + num_of_feature

            for p in range(0,num_of_feature):
                if features[n1_index,start+p] == 1 and features[n2_index,start+p] == 1:
                    count = count + 1
            whole_similarity[k,n+2] = whole_similarity[k,n+2] + 100*count/num_of_feature
        
        k = k + 1

for i in range(0,len(edges)):
    
    N1 = edges[i][0]
    N2 = edges[i][1]
    similarity[i][0] = N1
    similarity[i][1] = N2
    N1_index = int(np.where(nodes == N1)[0])
    N2_index = int(np.where(nodes == N2)[0])

    end = 1
    for n in range(0, len(featurelist)):
        count = 0
        start = end
        num_of_feature = feat_name.count(featurelist[n])
        end = start + num_of_feature
        
        for j in range(0,num_of_feature):
            if features[N1_index,start+j] == 1 and features[N2_index,start+j] == 1:
                count = count + 1

        similarity[i][n+2] = similarity[i][n+2] + 100*count/num_of_feature
                   

#- calculate similarity average
#- avg_similarity : percentage of contribution of feature to make an Edge

avg_similarity = np.zeros((1,19))

for i in range(0,19):
    avg_similarity[0,i] = np.average(similarity[:,i+2])

value = sum(avg_similarity[0,:])

for j in range(0,19):
    avg_similarity[0,j] = round(avg_similarity[0,j]/value,2)

### calculate edge weight from avg_similarity[0,:]
#- find TOP 5
#- TOP 5 : education_type 42.61 / gender 25.29 / locale 19.11 / location 2.93 / education_school 2.04

order = avg_similarity.argsort()
rank = order.argsort()
rank_of_avg_similarity = avg_similarity.shape[1]-rank

#- Set weight with normalization
r1 = np.where(rank_of_avg_similarity <= 5)[1][0]
r2 = np.where(rank_of_avg_similarity <= 5)[1][1]
r3 = np.where(rank_of_avg_similarity <= 5)[1][2]
r4 = np.where(rank_of_avg_similarity <= 5)[1][3]
r5 = np.where(rank_of_avg_similarity <= 5)[1][4]

for i in range(0,len(edges)):
    G.add_edge(similarity[i][0],similarity[i][1],
               weight = (similarity[i,r1+2]* avg_similarity[0,r1]                         
               + similarity[i,r2+2]*avg_similarity[0,r2]
               + similarity[i,r3+2]*avg_similarity[0,r3]
               + similarity[i,r4+2]*avg_similarity[0,r4]
               + similarity[i,r5+2]*avg_similarity[0,r5])/100)


#- calculate edge weight mean, deviation

edgeweight_list = np.zeros((1,len(edges)))
norm_list = np.zeros((1,len(edges)))

index = 0
for (u,v,weight) in G.edges.data('weight'):
    edgeweight_list[0][index] = edgeweight_list[0][index] + weight
    index = index + 1

edgeweight_mean = np.mean(edgeweight_list[0])
edgeweight_std = np.std(edgeweight_list[0])

edgeweight_gaussian = scale(edgeweight_list[0])
edgeweight_gaussian_mean = np.mean(edgeweight_gaussian)
edgeweight_gaussian_std = np.std(edgeweight_gaussian)

# Z > 0.52 means Upper 30% of edge weight
elarge = [(u,v) for (u,v,d) in G.edges(data=True) if (d['weight'] - edgeweight_mean)/edgeweight_std > 0.52]
elarge_weight = [d['weight'] for (u,v,d) in G.edges(data = True) if (d['weight'] - edgeweight_mean)/edgeweight_std > 0.52]
esmall = [(u,v) for (u,v,d) in G.edges(data=True) if (d['weight'] - edgeweight_mean)/edgeweight_std <= 0.52]

### if there is NO edge between node pairs, then calcuate "edge weight" value
### if edge weight value is more higher then make an edge and add edge weight
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

for k in range(0,len(NoEdgelist_index)):
    NoEdgeWeight[k,0] = NoEdgeSimilarity[k,0]
    NoEdgeWeight[k,1] = NoEdgeSimilarity[k,1]
    NoEdgeWeight[k,2] = (NoEdgeSimilarity[k,r1+2]* avg_similarity[0,r1]                         
               + NoEdgeSimilarity[k,r2+2]*avg_similarity[0,r2]
               + NoEdgeSimilarity[k,r3+2]*avg_similarity[0,r3]
               + NoEdgeSimilarity[k,r4+2]*avg_similarity[0,r4]
               + NoEdgeSimilarity[k,r5+2]*avg_similarity[0,r5])/100


# modify NoEdgeWeight value to Gaussian distribution
# if their value is greater than 1.28 :: more than 90%, add edge and add edge weight to graph

added_index = []
for i in range(0,len(NoEdgelist_index)):
    WeightCheck = (NoEdgeWeight[i,2] - edgeweight_mean)/edgeweight_std

    if (WeightCheck - edgeweight_gaussian_mean)/edgeweight_gaussian_std > 1.28:
        
        G.add_edge(NoEdgeWeight[i,0],NoEdgeWeight[i,1], weight = NoEdgeWeight[i,2])
        added_index.append(i)


### make adjacency matrix
adjacency_matrix = nx.adjacency_matrix(G)
adjacency_matrix = adjacency_matrix.todense()
adj_dataframe = pd.DataFrame(adjacency_matrix, index = nodes, columns = nodes)
adj_dataframe.to_csv('ego414_adjacencymatrix.csv')

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

# elarge : black edge
nx.draw_networkx_edges(G, pos, edgelist = elarge, width = 1,alpha = 1)
# esmall : blue dotted edge
nx.draw_networkx_edges(G, pos, edgelist = esmall, width = 1, alpha = 0.4, edge_color='b', style = 'dotted')
# added_large : green dotted edge
nx.draw_networkx_edges(G, pos, edgelist = added_large, width = 1, alpha = 0.7, edge_color = 'g', style = 'dotted')
# added_small : yellow dteed edge
nx.draw_networkx_edges(G, pos, edgelist = added_small, width = 1, alpha = 0.3, edge_color = 'y', style = 'dotted')

#nx.draw_neworkx_labels(G, pos, font_size = 10, font_family = 'sans-serif')

plt.axis('off')
plt.show()
