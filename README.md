# GraphClutering

Data is collected from http://snap.stanford.edu/data/egonets-Facebook.html
and then I use ego414 data for graph clustering

After doing pre-processing the data, with Jaccard Similarity I calculate the
weight of features of nodes if there exists edge.

Then calculate edge weight(node similarity) and apply this edge weight to
NEO-K-Means
