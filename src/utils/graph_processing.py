import pandas as pd
import networkx as nx
import numpy as np

def reduce_graph(G, degree_cutoff):
    G_copy = G.copy()
    
    degree_df = pd.DataFrame(
        dict(
            G_copy.degree()
        ), 
        index = ["Degree"]
    ).transpose()

    hub_criteria = degree_df["Degree"]>degree_cutoff
    
    hubs_deleted = list(
                degree_df[hub_criteria].reset_index()["index"]
            )

    G_copy.remove_nodes_from(
        hubs_deleted
    )        
    
    return(G_copy)

def graph_community(G, resolution=10): # large resolution to have smaller communities
    G_copy = G.copy()

    directed = nx.is_directed(G_copy)
    
    community_list = nx.community.louvain_communities(G_copy, resolution=resolution, seed=1997)

    # Create a dictionary to map nodes to their community
    node_community = {}
    for idx, community in enumerate(community_list):
        for node in community:
            node_community[node] = idx

    # Create a new graph with only intra-community edges
    if directed:
        H = nx.DiGraph()
    else:
        H = nx.Graph()
        
    H.add_nodes_from(G.nodes(data=True))  # Add all nodes with their attributes

    # Add only edges that connect nodes within the same community
    for u, v in G.edges():
        if node_community[u] == node_community[v]:
            H.add_edge(u, v, **G[u][v])
        
    return(H)