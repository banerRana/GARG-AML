import pandas as pd
import networkx as nx
import numpy as np

def measure_1_function(piece_1_dim, adj_full):
    piece_1 = adj_full[:piece_1_dim[0], :piece_1_dim[1]]
    total_sum_1 = piece_1.sum()
    total_size_1 = piece_1.size
    reduced_size_1 = total_size_1 - (3 * piece_1_dim[0]) + 2

    if reduced_size_1 > 0:
        rel_1 = total_sum_1/reduced_size_1
    else:
        rel_1 = 0
        
    return rel_1 

def measure_2_function(piece_1_dim, piece_2_dim, adj_full):
    piece_2 = adj_full[piece_1_dim[0]:piece_1_dim[0]+piece_2_dim[0], :piece_2_dim[1]]
    total_sum_2 = piece_2.sum()
    reduced_sum_2 = total_sum_2 - piece_2_dim[0]
    total_size_2 = piece_2.size
    reduced_size_2 = total_size_2 - piece_2_dim[0]

    if reduced_size_2 > 0:
        rel_2 = reduced_sum_2/reduced_size_2
    else:
        rel_2 = 0
        
    return rel_2

def measure_3_function(piece_1_dim, piece_2_dim, piece_3_dim, adj_full):
    piece_3 = adj_full[piece_1_dim[0]:, piece_2_dim[1]:]
    total_sum_3 = piece_3.sum()
    total_size_3 = piece_3.size
    reduced_size_3 = total_size_3 - piece_3_dim[0]

    if reduced_size_3 > 0:
        rel_3 = total_sum_3/reduced_size_3
    else:
        rel_3 = 0
    
    return rel_3

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

def summaries_neighbourhoors_node(node, G_copy, measures):
    G_ego = nx.ego_graph(G_copy, node)
    G_ego.remove_node(node)
    ego_list = list(G_ego.nodes())
    
    list_ego_measures = []
    for n in ego_list:
        list_ego_measures.append(measures[n])

    try:
        min_measure = np.min(list_ego_measures) 
        average_measure = np.mean(list_ego_measures)
        max_measure = np.max(list_ego_measures)

    except: 
        min_measure = 0
        average_measure = 0
        max_measure = 0
        
    return([min_measure, average_measure, max_measure])

def degree_neighbours_node(node, G_copy, G_degree_dict):
    G_ego = nx.ego_graph(G_copy, node)
    G_ego.remove_node(node)
    ego_list = list(G_ego.nodes())
    
    list_ego_degree = []
    for n in ego_list:
        list_ego_degree.append(G_degree_dict[n])

    try:
        min_degree = np.min(list_ego_degree)
        average_degree = np.mean(list_ego_degree)
        max_degree = np.max(list_ego_degree)

    except: 
        min_degree = 0
        average_degree = 0
        max_degree = 0
        
    return([min_degree, average_degree, max_degree])

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

def GARG_AML(G, degree_cutoff=10):
    G_reduced = reduce_graph(G, degree_cutoff)
    
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
    
    GARG_AML_df = combine_GARG_AML(G, GARG_AML_values, summaries_neighbours, degree_neighbours)
    
    return(GARG_AML_df)

def GARG_AML_nodeselection(G_ego_second, node):
  nodes_1 = list(nx.ego_graph(G_ego_second, node).nodes())
  nodes_1.remove(node)
  nodes_2 = list(G_ego_second.nodes)
  nodes_2.remove(node)
  for n in nodes_1:
      nodes_2.remove(n)

  nodes_ordered = [node]
  for n in nodes_2:
      nodes_ordered.append(n)
  for n in nodes_1:
      nodes_ordered.append(n)
      
  return nodes_1, nodes_2, nodes_ordered

def GARG_AML_node(node, G_degree_dict, G_copy):
    G_ego = nx.ego_graph(G_copy, node)

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
