import pandas as pd
import networkx as nx
import numpy as np

from .utils.neighbourhood_functions import summaries_neighbourhoors_node, degree_neighbours_node, GARG_AML_nodeselection
from .utils.measure_functions import measure_1_function, measure_2_function, measure_3_function

def GARG_AML_node(node, G_copy):
    G_ego_second = nx.ego_graph(G_copy, node, 2)
    
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
    G_degree_dict = dict(G_reduced.degree())
    
    GARG_AML_values = dict()
    
    nodes = list(G_reduced.nodes)
    for node in nodes:
        GARG_AML_node_value = GARG_AML_node(node, G_degree_dict, G_reduced)
        GARG_AML_values[node] = GARG_AML_node_value
        
    summaries_neighbours = dict()
    degree_neighbours = dict()
    
    for node in nodes:
        summaries_neighbours[node] = summaries_neighbourhoors_node(node, G_reduced, GARG_AML_values)
        degree_neighbours[node] = degree_neighbours_node(node, G_reduced, G_degree_dict)
    
    GARG_AML_df = combine_GARG_AML(G_reduced, GARG_AML_values, summaries_neighbours, degree_neighbours)
    
    return(GARG_AML_df)

