import networkx as nx
import scipy.io
# path = "dataset/archives/cond-mat/cond-mat.gml"
# path = "dataset/archives/celegansneural/celegansneural.gml"
# path = "dataset/archives/hep-th/hep-th.gml"

name = "nets"
path = f"../../dataset/data/{name}.gml"

# graph_matrix = scipy.io.mmread(path)
# G = nx.Graph(graph_matrix)

# Load the graph
G = nx.read_gml(path, label='id')

print("Number of nodes: ", G.number_of_nodes())
print("Number of edges: ", G.number_of_edges())

path_write = "../../dataset/data/"
# Write the graph in a gml file
# nx.write_gml(G, path_write + f"{name}.gml")
nx.write_edgelist(G, path_write + f"{name}.mtx", data=False)
