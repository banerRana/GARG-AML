import pandas as pd
import networkx as nx
import numpy as np

from .utils.measure_functions_undirected import measure_1_function, measure_2_function, measure_3_function
from .utils.measure_functions_directed import *
from .utils.neighbourhood_functions import summaries_neighbourhoors_node, degree_neighbours_node, GARG_AML_nodeselection

def GARG_AML_node_directed(node, G_copy, use_inverse = False):
    G_ego_second = nx.ego_graph(G_copy, node, 2, undirected=True) #Use both incoming and outgoing edges

    nodes_0, nodes_1, nodes_2, nodes_ordered = GARG_AML_nodeselection(G_ego_second, node)

    adj_full = nx.adjacency_matrix(G_ego_second, nodelist=nodes_ordered).toarray()

    size_0 = len(nodes_0)
    size_1 = len(nodes_1)
    size_2 = len(nodes_2)

    measure_00 = measure_00_function(adj_full, size_0)
    measure_01 = measure_01_function(adj_full, size_0, size_1)
    measure_02 = measure_02_function(adj_full, size_0, size_1, size_2)
    measure_10 = measure_10_function(adj_full, size_0, size_1)
    measure_11 = measure_11_function(adj_full, size_0, size_1)
    measure_12 = measure_12_function(adj_full, size_0, size_1, size_2)  
    measure_20 = measure_20_function(adj_full, size_0, size_2)
    measure_21 = measure_21_function(adj_full, size_0, size_1, size_2)
    measure_22 = measure_22_function(adj_full, size_2)

    measure_high = np.mean([measure_01, measure_12])
    measure_low = np.mean([measure_00, measure_02, measure_10, measure_11, measure_20, measure_21, measure_22])

    measure = measure_high - measure_low

    if use_inverse:
        measure = max(measure, -measure)
    
    return(measure)

def GARG_AML_node_undirected(node, G_copy):
    G_ego_second = nx.ego_graph(G_copy, node, 2)
    
    # nodes_ordered are the nodes ordered as node, 2nd order and 1st order neighbours
    nodes_1, nodes_2, nodes_ordered = GARG_AML_nodeselection(G_ego_second, node)
    
    adj_full = nx.adjacency_matrix(G_ego_second, nodelist=nodes_ordered).toarray()
    
    size_second = len(nodes_2)
    size_first = len(nodes_1)

    piece_1_dim = [size_second + 1, size_second + 1]
    piece_2_dim = [size_first, size_second + 1]
    piece_3_dim = [size_first, size_first]
    
    measure_1 = measure_1_function(piece_1_dim, adj_full)
    measure_2 = measure_2_function(piece_1_dim, piece_2_dim, adj_full)
    measure_3 = measure_3_function(piece_1_dim, piece_2_dim, piece_3_dim, adj_full)
    
    measure = measure_2 - (measure_1 + measure_3)/2

    return(measure)

def GARG_AML_node(node, G_copy, directed = False):
    if directed:
        return(GARG_AML_node_directed(node, G_copy))
    else:
        return(GARG_AML_node_undirected(node, G_copy))
    

def combine_GARG_AML(G_selection, measures_dict, summary_dict, neigh_degree_dict):
    degree_df = pd.DataFrame(
        dict(
            G_selection.degree()
        ), 
        index = ["Degree"]
    ).transpose()
    
    measures_df = pd.DataFrame(
        measures_dict, 
        index = ["GARGAML"]
    ).transpose()

    summary_df = pd.DataFrame(
        summary_dict, 
        index = ["ScoreMin", "ScoreMean", "ScoreMax"]
    ).transpose()

    neigh_degree_df = pd.DataFrame(
        neigh_degree_dict,
        index = ["DegMin", "DegMean", "DegMax"]
    ).transpose()
    
    GARG_AML_df = measures_df.merge(
        summary_df,
        left_index = True, 
        right_index = True
    ).merge(
        degree_df, 
        left_index = True, 
        right_index = True
    ).merge(
        neigh_degree_df,
        left_index = True, 
        right_index = True
    )
    
    return(GARG_AML_df)

def GARG_AML(G_reduced): # The method works with a pre-processed graph. G_reduced is the graph with the degree cutoff applied.
    directed = nx.is_directed(G_reduced)
    
    G_degree_dict = dict(G_reduced.degree())
    
    GARG_AML_values = dict()
    
    nodes = list(G_reduced.nodes)
    for node in nodes:
        GARG_AML_node_value = GARG_AML_node(node, G_reduced, directed=directed)
        GARG_AML_values[node] = GARG_AML_node_value
        
    summaries_neighbours = dict()
    degree_neighbours = dict()
    
    for node in nodes:
        summaries_neighbours[node] = summaries_neighbourhoors_node(node, G_reduced, GARG_AML_values)
        degree_neighbours[node] = degree_neighbours_node(node, G_reduced, G_degree_dict)
    
    GARG_AML_df = combine_GARG_AML(G_reduced, GARG_AML_values, summaries_neighbours, degree_neighbours)
    
    return(GARG_AML_df)

