# Types Hinting
from typing import List, Tuple, Annotated

# Graphs
import networkx as nx
import igraph as ig

# Plotting
import matplotlib.pyplot as plt
plt.style.use('default')

# Misc
from src.community_algs.configs import DetectionAlgorithms # Algs Names
import os



class DetectionAlgorithm(object):
    def __init__(self, alg_name: str) -> None:
        """
        Initialize the DetectionAlgorithm object
        
        Parameters
        ----------
        alg_name : str
            The name of the algorithm
        """
        self.alg_name = alg_name
        self.ig_graph = None

    def networkx_to_igraph(self, graph: nx.Graph) -> ig.Graph:
        """
        Convert networkx graph to igraph graph, in this way we can use 
        igraph's community detection algorithms
        
        Parameters
        ----------
        graph : nx.Graph
            The graph to be converted
        
        Returns
        ----------
        ig.Graph
            The converted graph
        """
        self.ig_graph = ig.Graph.from_networkx(graph)
        return self.ig_graph

    def compute_community(self, graph: nx.Graph, args: dict = None) -> List[List[int]]:
        """
        Compute the community detection algorithm
        
        Parameters
        ----------
        graph : nx.Graph
            The graph to be computed
        args : dict
            The arguments for the algorithm
        
        Returns
        ----------
        List[List[int]]
            list of list of vertices in each cluster
        """
        # Transform the graph to igraph
        graph = self.networkx_to_igraph(graph)

        # Rename DetectionAlgorithms Enum to da for convenience
        da = DetectionAlgorithms()
        # Choose the algorithm
        if self.alg_name == da.louv.value:
            return self.compute_louv(graph, args)
        elif self.alg_name == da.walk.value:
            return self.compute_walk(graph, args)
        elif self.alg_name == da.gre.value:
            return self.compute_gre(graph, args)
        elif self.alg_name == da.inf.value:
            return self.compute_inf(graph, args)
        elif self.alg_name == da.lab.value:
            return self.compute_lab(graph, args)
        elif self.alg_name == da.eig.value:
            return self.compute_eig(graph, args)
        elif self.alg_name == da.btw.value:
            return self.compute_btw(graph, args)
        elif self.alg_name == da.spin.value:
            return self.compute_spin(graph, args)
        elif self.alg_name == da.opt.value:
            return self.compute_opt(graph, args)
        elif self.alg_name == da.scd.value:
            return self.compute_scd(graph, args)
        else:
            raise ValueError('Invalid algorithm name')

    def vertexcluster_to_list(self, cluster: nx.VertexClustering) -> List[List[int]]:
        """
        Convert igraph.VertexClustering object to list of list of vertices in each cluster

        Parameters
        ----------
        cluster : VertexClustering
            cluster from igraph community detection algorithm

        Returns
        -------
        List[List[int]]
            list of list of vertices in each cluster
        """
        return [c for c in cluster]

    def plot_graph(self) -> plt:
        """Plot the graph using igraph
        
        Returns
        ---------
        plot: plt
            The plot of the graph
        
        """
        # fig, ax = plt.subplots(figsize=(10, 10))
        plot = ig.plot(
            self.ig_graph,
            mark_groups=True,
            vertex_size=20,
            edge_color='black',
            vertex_label=[v.index for v in self.ig_graph.vs],
            bbox=(0, 0, 500, 500),
            # target=ax,
        )
        return plot

    def compute_louv(self, graph: ig.Graph, args_louv: dict) -> List[List[int]]:
        """
        Compute the Louvain community detection algorithm
        
        Parameters
        ----------
        graph : ig.Graph
            The graph to be clustered
        args_louv : dict
            The arguments for the Louvain algorithm

        Returns
        ----------
        List[List[int]]
            list of list of vertices in each cluster
        """
        louv = graph.community_leiden(**args_louv)
        return self.vertexcluster_to_list(louv)

    def compute_walk(self, graph: ig.Graph, args_walk: dict) -> List[List[int]]:
        """
        Compute the Walktrap community detection algorithm
        
        Parameters
        ----------
        graph : ig.Graph
            The graph to be clustered
        args_walk : dict
            The arguments for the Walktrap algorithm

        Returns
        ----------
        List[List[int]]
            list of list of vertices in each cluster
        """
        walk = graph.community_walktrap(**args_walk)
        # Need to be converted to VertexClustering object
        return self.vertexcluster_to_list(walk.as_clustering())

    def compute_gre(self, graph: ig.Graph, args_gre: dict) -> List[List[int]]:
        """
        Compute the Greedy community detection algorithm
        
        Parameters
        ----------
        graph : ig.Graph
            The graph to be clustered
        args_greed : dict
            The arguments for the Greedy algorithm

        Returns
        ----------
        List[List[int]]
            list of list of vertices in each cluster
        """
        greed = graph.community_fastgreedy(**args_gre)
        # Need to be converted to VertexClustering object
        return self.vertexcluster_to_list(greed.as_clustering())

    def compute_infomap(self, graph: ig.Graph, args_infomap: dict) -> List[List[int]]:
        """
        Compute the Infomap community detection algorithm
        
        Parameters
        ----------
        graph : ig.Graph
            The graph to be clustered
        args_infomap : dict
            The arguments for the Infomap algorithm

        Returns
        ----------
        List[List[int]]
            list of list of vertices in each cluster
        """
        infomap = graph.community_infomap(**args_infomap)
        return self.vertexcluster_to_list(infomap)

    def compute_lab(self, graph: ig.Graph, args_lab: dict) -> List[List[int]]:
        """
        Compute the Label Propagation community detection algorithm
        
        Parameters
        ----------
        graph : ig.Graph
            The graph to be clustered
        args_lab : dict
            The arguments for the Label Propagation algorithm

        Returns
        ----------
        List[List[int]]
            list of list of vertices in each cluster
        """
        lab = graph.community_label_propagation(**args_lab)
        return self.vertexcluster_to_list(lab)

    def compute_eig(self, graph: ig.Graph, args_eig: dict) -> List[List[int]]:
        """
        Compute the Eigenvector community detection algorithm
        
        Parameters
        ----------
        graph : ig.Graph
            The graph to be clustered
        args_eig : dict
            The arguments for the Eigenvector algorithm

        Returns
        ----------
        List[List[int]]
            list of list of vertices in each cluster
        """
        eig = graph.community_leading_eigenvector(**args_eig)
        return self.vertexcluster_to_list(eig)

    def compute_btw(self, graph: ig.Graph, args_btw: dict) -> List[List[int]]:
        """
        Compute the Edge Betweenness community detection algorithm
        
        Parameters
        ----------
        graph : ig.Graph
            The graph to be clustered
        args_btw : dict
            The arguments for the Betweenness algorithm

        Returns
        ----------
        List[List[int]]
            list of list of vertices in each cluster
        """
        btw = graph.community_edge_betweenness(**args_btw)
        # Need to be converted to VertexClustering object
        return self.vertexcluster_to_list(btw.as_clustering())

    def compute_spin(self, graph: ig.Graph, args_spin: dict) -> List[List[int]]:
        """
        Compute the Spin Glass community detection algorithm
        
        Parameters
        ----------
        graph : ig.Graph
            The graph to be clustered
        args_spin : dict
            The arguments for the Spin Glass algorithm

        Returns
        ----------
        List[List[int]]
            list of list of vertices in each cluster
        """
        spin = graph.community_spinglass(**args_spin)
        return self.vertexcluster_to_list(spin)

    def compute_opt(self, graph: ig.Graph, args_opt: dict) -> List[List[int]]:
        """
        Compute the Optimal community detection algorithm
        
        Parameters
        ----------
        graph : ig.Graph
            The graph to be clustered
        args_opt : dict
            The arguments for the Optimal algorithm

        Returns
        ----------
        List[List[int]]
            list of list of vertices in each cluster
        """
        opt = graph.community_optimal_modularity(**args_opt)
        return self.vertexcluster_to_list(opt)

    def compute_scd(self, graph: ig.Graph, args_scd: dict) -> List[List[int]]:
        """
        Compute the Surprise community detection algorithm
        
        Parameters
        ----------
        graph : ig.Graph
            The graph to be clustered
        args_scd : dict
            The arguments for the Surprise algorithm

        Returns
        ----------
        List[List[int]]
            list of list of vertices in each cluster
        """
        # Write the graph to a text file
        self.write_graph_to_file(graph, "output.txt")
        # Execute SCD algorithm from the git submodule
        os.system("./../src/SCD/build/scd -f output.txt")
        result_list = self.read_data_from_file('communities.dat')
        return result_list

    @staticmethod
    def write_graph_to_file(graph: ig.Graph, file_path: str) -> None:
        """
        Write the graph to a text file, where each line is an 
        edge in the graph.

        Parameters
        ----------
        graph : ig.Graph
            Graph object to write to file
        file_path : str
            file path of the output file
        """
        with open(file_path, 'w') as file:
            for edge in graph.get_edgelist():
                # To ensure we don't duplicate edges (x, y) and (y, x)
                if edge[0] < edge[1]:
                    file.write(f"{edge[0]} {edge[1]}\n")

    @staticmethod
    def read_data_from_file(file_path: str) -> List[List[int]]:
        """
        Read data from file and return a list of lists, where each row list of
        nodes is a community.

        Parameters
        ----------
        file_path : str
            File path to the data file.

        Returns
        -------
        List[List[int]]
            List of lists, where each row list of nodes is a community.
        """
        data_list = []
        with open(file_path, 'r') as file:
            for line in file:
                numbers = [int(num) for num in line.strip().split()]
                data_list.append(numbers)
        return data_list