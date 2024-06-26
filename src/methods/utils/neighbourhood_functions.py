import networkx as nx
import numpy as np

def summaries_neighbourhoors_node(node, G_copy, measures):
    G_ego = nx.ego_graph(G_copy, node)
    G_ego.remove_node(node)
    ego_list = list(G_ego.nodes)
    
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
    ego_list = list(G_ego.nodes)
    
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

def GARG_AML_nodeselection_undirected(G_ego_second, node):
    nodes_1 = list(
      nx.ego_graph(G_ego_second, node).nodes)
    nodes_1.remove(node)
    nodes_2 = list(G_ego_second.nodes)
    nodes_2.remove(node)
    for n in nodes_1:
        nodes_2.remove(n)

    # For undirected networks, specific order to obtain scores (group node with second order neighbours)
    nodes_ordered = [node] + nodes_2 + nodes_1
        
    return nodes_1, nodes_2, nodes_ordered

def GARG_AML_nodeselection_directed(G_ego_second, node):
    nodes_1 = list(
      nx.ego_graph(G_ego_second, node, undirected=True).nodes
      )
    nodes_1.remove(node)
    nodes_2 = list(G_ego_second.nodes)

    nodes_2_s = list(
        nx.ego_graph(G_ego_second, node, undirected=False, radius=2).nodes
        )
    
    nodes_0 = list(
        set(nodes_2).difference(set(nodes_2_s)).difference(set(nodes_1))
    )

    nodes_0 = [node] + nodes_0

    for n in nodes_0:
        nodes_2.remove(n)
    for n in nodes_1:
        nodes_2.remove(n)

    # For directed network, specific order to obtain scores (in order of "group")
    nodes_ordered = nodes_0 + nodes_1 + nodes_2
        
    return nodes_0, nodes_1, nodes_2, nodes_ordered

def GARG_AML_nodeselection(G_ego_second, node):
    if G_ego_second.is_directed():
        return GARG_AML_nodeselection_directed(G_ego_second, node)
    else:
        return GARG_AML_nodeselection_undirected(G_ego_second, node)