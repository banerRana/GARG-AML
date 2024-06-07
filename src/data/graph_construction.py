import pandas as pd
import networkx as nx

def construct_IBM_graph(path="data/HI-Small_Trans.csv"):
    """
    Construct a graph from the IBM data.
    """
    # Load the data
    data = pd.read_csv(path)
    # Create the graph
    G = nx.Graph()
    
    edges = zip(data["Account"], data["Account.1"])

    G.add_edges_from(edges)

    return G