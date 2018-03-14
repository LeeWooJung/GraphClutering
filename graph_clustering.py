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

### make graph using NetworkX

G = nx.Graph()
nodes = features[:,0]
attributes = features[:,1:]

#- ADD node and node's attributes
for i in range(0,len(nodes)):
    G.add_node(nodes[i],attr = attributes[i])

#- Add edge
G.add_edges_from(edges)

### calculate jaccard similarity if there's an edge

#- note : feature names are (19)
#-      : birthday / education_concentration / education_degree / education_school / education_type / education_year
#-      : first_name / gender / hometown / languages / last_name / locale / location / name / work_employer
#-      : work_end_date / work_location / work_position / work_start_date

edges = list(G.edges)

similarity = np.zeros((len(edges),2+19))

for i in range(0,len(edges)):
    
    N1 = edges[i][0]
    N2 = edges[i][1]
    similarity[i][0] = N1
    similarity[i][1] = N2
    N1_index = int(np.where(nodes == N1)[0])
    N2_index = int(np.where(nodes == N2)[0])

    count = 0
    start = 1
    num_of_feature = feat_name.count('birthday')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][2] = similarity[i][2]+100*count/num_of_feature
    
    count = 0
    start = end
    num_of_feature = feat_name.count('education_concentration')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][3] = similarity[i][3]+100*count/num_of_feature

    count = 0
    start = end
    num_of_feature = feat_name.count('education_degree')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][4] = similarity[i][4]+100*count/num_of_feature

    count = 0
    start = end
    num_of_feature = feat_name.count('education_school')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][5] = similarity[i][5]+100*count/num_of_feature
    

    count = 0
    start = end
    num_of_feature = feat_name.count('education_type')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][6] = similarity[i][6]+100*count/num_of_feature

    count = 0
    start = end
    num_of_feature = feat_name.count('education_year')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][7] = similarity[i][7]+100*count/num_of_feature

    count = 0
    start = end
    num_of_feature = feat_name.count('first_name')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][8] = similarity[i][8]+100*count/num_of_feature

    count = 0
    start = end
    num_of_feature = feat_name.count('gender')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][9] = similarity[i][9]+100*count/num_of_feature

    count = 0
    start = end
    num_of_feature = feat_name.count('hometown')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][10] = similarity[i][10]+100*count/num_of_feature

    count = 0
    start = end
    num_of_feature = feat_name.count('languages')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][11] = similarity[i][11]+100*count/num_of_feature

    count = 0
    start = end
    num_of_feature = feat_name.count('last_name')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][12] = similarity[i][12]+100*count/num_of_feature

    count = 0
    start = end
    num_of_feature = feat_name.count('locale')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][13] = similarity[i][13]+100*count/num_of_feature

    count = 0
    start = end
    num_of_feature = feat_name.count('location')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][14] = similarity[i][14]+100*count/num_of_feature

    count = 0
    start = end
    num_of_feature = feat_name.count('name')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][15] = similarity[i][15]+100*count/num_of_feature

    count = 0
    start = end
    num_of_feature = feat_name.count('work_employer')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][16] = similarity[i][16]+100*count/num_of_feature

    count = 0
    start = end
    num_of_feature = feat_name.count('work_end_date')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][17] = similarity[i][17]+100*count/num_of_feature
    
    count = 0
    start = end
    num_of_feature = feat_name.count('work_location')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][18] = similarity[i][18]+100*count/num_of_feature

    count = 0
    start = end
    num_of_feature = feat_name.count('work_position')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][19] = similarity[i][19]+100*count/num_of_feature

    count = 0
    start = end
    num_of_feature = feat_name.count('work_start_date')
    end = start + num_of_feature
    for j in range(0,num_of_feature):
        if features[N1_index,start+j] == 1:
            if features[N2_index,start+j] == 1:
                count = count + 1
                
    similarity[i][20] = similarity[i][20]+100*count/num_of_feature
