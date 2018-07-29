"""
    The Author : LEE WOO JUNG in SungKyunKwan University
    Graph Clustering with attributes similarity
    Calculate the similarity of each node of Graph,
    then predict which pair of nodes can make edge later
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import csv

#%% Data Preprocessing

def makefile_414featurenamestxt():
    
    ego414 = np.genfromtxt('414.featnames', dtype = str, delimiter = "\n")
    egofeature = []
    
    file = open("414.featurenames.txt",'w')
    for i in range(0, len(ego414)):
        egofeature.extend(ego414[i].split(';anonymized feature ')[0:-1])
        if (egofeature[i][-3:] == ';id'):
            egofeature[i] = egofeature[i].strip(';id')
        egofeature[i] = egofeature[i].replace(';','_')
        line = str(egofeature[i]) + '\n'
        file.write(line)
    file.close()

def makefile_414featnamestxt():
    
    ego_414 = np.genfromtxt('414.featurenames.txt', dtype = str, delimiter = "\n")
    featnames = []
    for j in range(0,len(ego_414)):
        featnames.append((ego_414[j].split())[1])
    file = open("414featname.txt", 'w')
    for i in range(0,len(featnames)):
        if i == len(featnames) - 1:
            line = str(featnames[i])
        else:
            line = str(featnames[i]) + '\n'
        file.write(line)
    file.close()

#%% Make Feature Name list
    
def MakeFeatNamelist():
    file = open('414featnames.txt','r')
    feat_name = file.read()
    feat_name = feat_name.split('\n')
    feat_name.insert(0,'nodes')
    file.close()
    return feat_name

#%% Make Feature list

def MakeFeaturelist(feat_name):
    indexes = np.unique(feat_name, return_index = True)[1]
    featurelist = [feat_name[index] for index in sorted(indexes)]
    featurelist.remove('nodes')
    
    return featurelist
    

#%% Make Features

def FEATURES(feat_name):
    features = []
    feature = np.genfromtxt('414.feat', dtype = str, delimiter = "\n")
    features = np.zeros((len(feature), len(feat_name)))
    for i in range(0,len(feature)):
        features[i] = feature[i].split(' ')
    features = features.astype('int32')
    return features

#%% Read 414.edge : edges
    
def EDGE():
    edge = np.genfromtxt('414.edges', dtype = str, delimiter = "\n")
    edges = np.zeros((len(edge),2))
    for j in range(0,len(edge)):
        edges[j] = edge[j].split(' ')
    edges = edges.astype('int32')
    return edges

#%% Make Circle File

def make_circlefile(nodes):
    
    circle414 = np.genfromtxt('414.circles', dtype = str, delimiter = "\n")
    cluster = []
    cluster_ele = []
    cluster_total = []
    outlier = np.zeros((1,len(nodes)))
    
    file = open('414circle.txt','w')
    for i in range(0,len(circle414)):
        cluster.append(circle414[i].split('\t')[0][-1])
        for j in range(1,len(circle414[i].split('\t'))):
            cluster_ele.append(circle414[i].split('\t')[j])
            cluster_total.append(circle414[i].split('\t')[j])
        for k in range(0,len(cluster_ele)):
            if k == len(cluster_ele)-1:
                file.write(cluster_ele[k] + '\n')
            else:
                file.write(cluster_ele[k] + ' ')
                
        cluster_ele = []
    file.close()
    
    return cluster, cluster_total, outlier

#%% Make Cluster Matrix

def MakeClusterMatrix(NumOfNodes, NumOfClusters):
    clustermatrix = np.zeros((NumOfNodes, NumOfClusters))
    f = open('414circle.txt','r')
    line = f.readline()
    i = 0
    while line:
        line = line.split(' ')
        line[len(line)-1] = line[len(line)-1][:-1]
        for j in range(0,len(line)):
            clustermatrix[np.where(nodes == int(line[j]))[0][0],i] += 1
        line = f.readline()
        i = i+1
    np.savetxt('original_cluster.csv', clustermatrix, fmt = '%d', delimiter = ',')
    f.close()
    return clustermatrix

#%% Add nodes & nodes' attributes AND edges to Graph

def Add_Nodes(G, nodes, attributes):
    for i in range(0,len(nodes)):
        G.add_node(nodes[i], attr = attributes[i])
    return G

def Add_Edges(G, edges):
    G.add_edges_from(edges)
    
    return G

#%% Find Ground Truth Beta Value
    
def FindGroundTruth(cluster_total, outlier, nodes):
    # ground truth alpha value
    alpha = len(cluster_total)/len(nodes) - 1
    
    # if outlier value is 0, it means that node is outlier
    for i in range(0,len(cluster_total)):
        outlier[0][np.where(nodes == int(cluster_total[i]))[0][0]] += 1
    beta = len(np.where(outlier == 0)) / len(nodes)
    return alpha, beta

#%% Make an original adjacency matrix

def MakeAdjacencyMatrix(nodes, edges_of_G):
    
    original_adj = np.zeros((len(nodes), len(nodes)))
    for (i,j) in edges_of_G:
        original_adj[np.where(nodes == i)[0][0], np.where(nodes == j)[0][0]] += 1
        original_adj[np.where(nodes == j)[0][0], np.where(nodes == i)[0][0]] += 1
    return original_adj

#%% Find what detailed features that have more than 90% jaccard similarity

def AlmostSameIndex(features):
    simindex = np.zeros((2,1))
    for i in range(1, len(features[0])-1):
        for j in range(i+1, len(features[0])):
            
            M01 = 0
            M10 = 0
            M11 = 0
            
            for k in range(0,len(features[:,i])):
                if features[:,i][k] == 0 and features[:,j][k] == 1:
                    M01 += 1
                elif features[:,i][k] == 1 and features[:,j][k] == 0:
                    M10 += 1
                elif features[:,i][k] == 1 and features[:,j][k] == 1:
                    M11 += 1
            sim = M11 / (M01 + M10 + M11)
            if sim > 0.9:
                string = feat_name[i] + " is similar with " + feat_name[j] + " and feature indexes are " + str(i) + "," + str(j) + " and number of features is "
                value1 = feat_name.count(feat_name[i])
                value2 = feat_name.count(feat_name[j])
                PrintOut(string, value1, value2)
                
                simindex = np.c_[simindex, np.array([i,j])]
    return simindex

#%% Pop one column since two columns are redundant
    
def RemoveRedundantFeature(feat_name, featurlist, features, simindex):
    
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
    
    return features

#%% Calculate Jaccard Similarity

def Cal_JaccardSimilarity(edges_of_G, features, nodes, feat_name):
    
    edges = list(edges_of_G)
    #similarity = np.zeros((len(edges), 1+len(features[0])))
    #nodeC2 = int(len(nodes)*(len(nodes)-1)/2)
    #whole_similarity = np.zeros((nodeC2, 1+len(features[0])))
    
    onlyEdge = np.zeros((1,len(features[0])-1))
    whole = np.zeros((1,len(features[0])-1))
    
    for n1_index in range(0,len(nodes)):
        for n2_index in range(n1_index + 1, len(nodes)):
            for k in range(1,len(feat_name)):
                if features[n1_index][k] == 1 and features[n2_index][k] == 1:
                    whole[0][k-1] += 1
                    
    for i in range(len(edges)):
        N1 = edges[i][0]
        N2 = edges[i][1]
        n1_index = int(np.where(nodes == N1)[0])
        n2_index = int(np.where(nodes == N2)[0])
        
        for k in range(1,len(feat_name)):
            if features[n1_index][k] == 1 and features[n2_index][k] == 1:
                onlyEdge[0][k-1] += 1
                
    return onlyEdge, whole

#%% Calculate Ratio of similarity
    
def FindSimilarityRatio(features, whole, onlyEdge):
    
    ratio_whole = np.zeros((1,len(features[0])-1))
    ratio_onlyEdge = np.zeros((1,len(features[0])-1))
    
    sum_whole = sum(whole[0])
    sum_onlyEdge = sum(onlyEdge[0])
    
    for i in range(0,len(whole[0])):
        ratio_whole[0][i] += round((whole[0][i]/sum_whole)*100,5)
    for j in range(0,len(onlyEdge[0])):
        ratio_onlyEdge[0][j] += round((onlyEdge[0][j]/sum_onlyEdge)*100, 5)
    
    return ratio_whole, ratio_onlyEdge

def MoreSimilarFeatures(ratio_onlyEdge, ratio_whole, whole):
    
    upperindex = []
    uppervalue = []
    
    for k in range(0,len(whole[0])):
        if ratio_onlyEdge[0][k] - ratio_whole[0][k] > 0:
            upperindex.append(k)
            uppervalue.append(ratio_onlyEdge[0][k] - ratio_whole[0][k])
            
    return upperindex, uppervalue

def WriteToCsvFile_UpperFeatures(upperindex, uppervalue):
    
    upperfile = open('UpperIndexValue.csv','w',newline='')
    wr = csv.writer(upperfile)
    for i in range(0,len(upperindex)):
        wr.writerow([upperindex[i], uppervalue[i]])
    upperfile.close()

#%% Take upper 2% value of uppervalue

def Upper25(uppervalue, nodes, edges_of_G):
    
    edges = list(edges_of_G)
    nodesC2 = int(len(nodes)*(len(nodes)-1)/2)
    similarity = np.zeros((len(edges), 1 + len(features[0])))
    whole_similarity = np.zeros((nodesC2, 1 + len(features[0])))
    
    uppervalue_upper25_index = []
    uppervalue_upper25_value = sorted(uppervalue, reverse = True) ## sorting descending order
    uppervalue_upper25_value = uppervalue_upper25_value[0:round(0.25*len(uppervalue))] ## because of upper 25%
    
    for i in range(0,len(uppervalue_upper25_value)):
        uppervalue_upper25_index.append(upperindex[np.where(uppervalue_upper25_value[i] == uppervalue)[0][0]])
    
    uppervalue_norm = []
    
    for i in range(0,len(uppervalue_upper25_index)):
        uppervalue_norm.append(uppervalue_upper25_value[i]/sum(uppervalue_upper25_value))
    
    k = 0
    for n1_index in range(0,len(nodes)):
        for n2_index in range(n1_index+1, len(nodes)):
            n1 = nodes[n1_index]
            n2 = nodes[n2_index]
            whole_similarity[k,0] += n1
            whole_similarity[k,1] += n2
            
            for n in range(0,len(uppervalue_upper25_index)):
                if features[n1_index, uppervalue_upper25_index[n] + 1] == 1 and features[n2_index,uppervalue_upper25_index[n] + 1] == 1:
                    whole_similarity[k][uppervalue_upper25_index[n] + 2] += uppervalue_norm[n]
            k += 1
            
    for i in range(0,len(edges)):
        
        N1 = edges[i][0]
        N2 = edges[i][1]
        similarity[i][0] = N1
        similarity[i][1] = N2
        N1_index = int(np.where(nodes == N1)[0])
        N2_index = int(np.where(nodes == N2)[0])
        
        for n in range(0,len(uppervalue_upper25_index)):
            
            if features[N1_index, uppervalue_upper25_index[n] + 1] == 1 and features[N2_index, uppervalue_upper25_index[n] + 1] == 1:
                similarity[i][uppervalue_upper25_index[n] + 2] += uppervalue_norm[n]
    
    return similarity, whole_similarity, edges

#%% ADD Edge of Similar Node pairs which don't have edge
    
def BasicEdgeWeight(G, similarity, edges):
    
    for i in range(0,len(edges)):
        G.add_edge(similarity[i][0], similarity[i][1], weight = 1) ## 1: Edge Similarity[i][2:]
    
    return G

def NoEdgePairSimilarity(similarity, whole_similarity):
    
    NoEdgelist_index = []
    simlist = []
    
    for i in range(0,len(similarity)):
        simlist.append(list(similarity[i,0:2]))
    for j in range(0,len(whole_similarity)):
        if list(whole_similarity[j,0:2]) not in simlist:
            NoEdgelist_index.append(j)
    
    NoEdgeSimilarity = np.zeros((len(NoEdgelist_index), len(whole_similarity[0])))
    
    # Make array "NoEdgeSimilarity" which means similarity of node pair that have no edge
    for k in range(0,len(NoEdgelist_index)):
        NoEdgeSimilarity[k,:] = whole_similarity[NoEdgelist_index[k],:]
    
    # Assume there exist all edges for NO edge node pair,
    # then calculate edge weight
    NoEdgeWeight = np.zeros((len(NoEdgelist_index),3))
    
    for l in range(0,len(NoEdgelist_index)):
        NoEdgeWeight[l,0] = NoEdgeSimilarity[l,0]
        NoEdgeWeight[l,1] = NoEdgeSimilarity[l,1]
        NoEdgeWeight[l,2] = sum(NoEdgeSimilarity[l,2:])
        
    # Modify NoEdgeWeight value to Gaussian Distribution
    # Upper 32%(upper average), add edge and add weight to graph

    NWE = sorted(NoEdgeWeight[:,2], reverse = True)
    
    return NoEdgeWeight , NWE

def AddEdge(NWE, NoEdgeWeight, G):
    
    NWE_truncated = NWE[0:round(0.32*len(NWE))] ## because of upper 31%
    NWE_index = []
    added_index = []
    
    for i in range(0,len(NWE_truncated)):
        NWE_index.append(np.where(NWE_truncated[i] == NoEdgeWeight[:,2])[0][0])
    
    for j in range(0,len(NWE_index)):
        G.add_edge(NoEdgeWeight[NWE_index[j],0], NoEdgeWeight[NWE_index[j],1], weight = 1)
        added_index.append(j)
        
    return NWE_truncated, NWE_index, added_index, G

#%% Make Adjacency Matrix & Write to File

def MakeAddedAdjacencyMatrix(G, nodes):
    adjacency_matrix = nx.adjacency_matrix(G, nodelist = nodes, weight = 'weight')
    adjacency_matrix = adjacency_matrix.todense()
    adj_dataframe = pd.DataFrame(adjacency_matrix, index = nodes, columns = nodes)
    adj_dataframe.to_csv('ego414_adjacencymatrix.csv')
    
    # make adjacency matrix about added edge, weighted graph
    
    added_edge_adj = np.zeros((len(nodes),len(nodes)))
    
    for (i,j) in G.edges:
        added_edge_adj[np.where(nodes == i)[0][0], np.where(nodes == j)[0][0]] += 1
        added_edge_adj[np.where(nodes == j)[0][0], np.where(nodes == i)[0][0]] += 1
    
    # adjacency matrix to csv file
    np.savetxt("Final_32%edge_added_adj.csv", added_edge_adj, fmt = '%d', delimiter = ",")
    
    return added_edge_adj
        
#%% PLOT
    
def PLOT_NoEdge(NoEdgeWeight):
    
    NWE_xaxis = []
    NWE_ascendingorder = sorted(NoEdgeWeight[:,2])
    
    for i in range(0,len(NWE_ascendingorder)):
        NWE_xaxis.append(i)
    
    plt.scatter(NWE_xaxis, NWE_ascendingorder, s = 5, marker = 'o')
    plt.xlabel('index')
    plt.ylabel('NoEdgeNodePairSimilairty')
    plt.title('Ascending Order Of NoEdge Node Pair Similairty')
    plt.show()
    
def PLOT_FinalGraph(G, edges, added_index, NoEdgeWeight):
    added = np.zeros((len(added_index),3))
    for i in range(0,len(added_index)):
        added[i,0] = NoEdgeWeight[added_index[i],0]
        added[i,1] = NoEdgeWeight[added_index[i],1]
        added[i,2] = NoEdgeWeight[added_index[i],2]
    
    added_large = []
    added_small = []
    
    for j in range(0,len(added)):
        if added[j,2] > np.mean(added[:,2]):
            added_large.append((added[j,0],added[j,1]))
        else:
            added_small.append((added[j,0],added[j,1]))
    
    pos = nx.spring_layout(G,dim = 2)
    nx.draw_networkx_nodes(G, pos, node_size = 1)
    nx.draw_networkx_edges(G, pos, edgelist = edges, width = 1, alpha = 1)
    nx.draw_networkx_edges(G, pos, edgelist = added_large, width = 1, alpha = 0.3, edge_color = 'b', style = 'dotted')
    nx.draw_networkx_edges(G, pos, edgelist = added_small, width = 1, alpha = 0.3, edge_color = 'g', style = 'dotted')
    
    plt.axis('off')
    plt.show()
    
#%% PrintOut
    
def PrintOut(string, *args):
    Val = []
    for value in args:
        Val.append(value)
    
    print(string + str(Val))

#%% Main Function   
if __name__ == "__main__":
    
    ## Data PreProcessing
    makefile_414featurenamestxt()
    makefile_414featnamestxt()
    
        #- make feat_name list : feat_name
    feat_name = MakeFeatNamelist()    
        #- features : node no. , node attribute to array
    features = FEATURES(feat_name)    
        #- read 414.edge : edges
    edges = EDGE()
    
    ## make graph using NetworkX
    G = nx.Graph()
    nodes = features[:,0]
    attributes = features[:,1:]    
        #- Add node & node's attributes to Graph
    G = Add_Nodes(G, nodes, attributes)
    G = Add_Edges(G, edges)    
        #- Check Ground truth value of the Graph
    cluster, cluster_total, outlier = make_circlefile(nodes)
        #- Make Ground Truth Cluster Matrix
    NumOfNodes = len(nodes)
    NumOfClusters = len(cluster)
    clustermatrix = MakeClusterMatrix(NumOfNodes, NumOfClusters)
        #- ground truth cluster number
    PrintOut('Ground Truth cluster number is ', NumOfClusters)
        #- find ground truth alpha, beta value
    alpha, beta = FindGroundTruth(cluster_total, outlier, nodes)
    PrintOut('Ground Truth alpha value is ', alpha)
    PrintOut('Ground Truth beta value is ', beta)
        #- make an adjacency matrix about original graph
    edges_of_G = G.edges
    original_adj = MakeAdjacencyMatrix(nodes, edges_of_G)
    np.savetxt("original_adj.csv", original_adj, fmt = "%d", delimiter = ",")
    
    PrintOut('original feature length is : ', len(features[0])-1)
    
    ## Find what detailed features that have more than 90% jaccard similarity
    simindex = AlmostSameIndex(features)
        #- Make Featurelist
    featurelist = MakeFeaturelist(feat_name)
        #- Pop one column since Simlar two columns ar redundant
    features = RemoveRedundantFeature(feat_name, featurelist, features, simindex)
    
    ## Calculate jaccrad similarity if there's an edge    
    onlyEdge, whole = Cal_JaccardSimilarity(edges_of_G, features, nodes, feat_name)
    
    ## What features have more simliarity if there's an edge on node pairs
        #- Calculate Ratio
    ratio_whole, ratio_onlyEdge = FindSimilarityRatio(features, whole, onlyEdge)
        #- Find features which have more similarity if there's an edge
    upperindex, uppervalue = MoreSimilarFeatures(ratio_onlyEdge, ratio_whole, whole)
        #- write to csv file
    WriteToCsvFile_UpperFeatures(upperindex, uppervalue)
    
    ## Take upper 25% value of uppervalue, because upper 25% occupy 93% of influences
    similarity, whole_similarity, edges = Upper25(uppervalue, nodes, edges_of_G)
    
    ## Between two nodes, if there's no edge and they have larger similarity,
    ## then add an edge between them, since they are very similar!
    
    G = BasicEdgeWeight(G, similarity, edges)
    NoEdgeWeight, NWE = NoEdgePairSimilarity(similarity, whole_similarity)
        #- Plot the similarity of Node Pair which have NO EDGE
    PLOT_NoEdge(NoEdgeWeight)
        #- if there is NO edge between node pairs, then calculate "edge weight" value
        #- if edge weight value is more larger, then make an edge and edge weight
    NWE_truncated, NWE_index, added_index, G = AddEdge(NWE, NoEdgeWeight, G)
    
    ## Make an adjacency Matrix
    added_edge_adj = MakeAddedAdjacencyMatrix(G, nodes)
    
    ## Plot Graph
    PLOT_FinalGraph(G, edges, added_index, NoEdgeWeight)


