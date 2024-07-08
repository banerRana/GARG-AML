import pandas as pd
import networkx as nx

def construct_IBM_graph(path="data/HI-Small_Trans.csv", directed=False):
    """
    Construct a graph from the IBM data.
    """
    # Load the data
    data = pd.read_csv(path)
    
    # Create the graph
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    edges = zip(data["Account"], data["Account.1"])

    G.add_edges_from(edges)
    G.remove_edges_from([(n, n) for n in G.nodes() if G.has_edge(n, n)])

    return G