import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import random

def add_smurfing_patterns_separate(graph, n):
    for i in range(n):
        # Add smurfing pattern to the graph

        # Determine number of nodes in the graph
        num_nodes = graph.vcount()

        # Determine number of nodes in the smurfing pattern
        num_nodes_mules = random.randint(2, 5) # Randomly have two to five money mules
        for i in range(num_nodes_mules+2):
            graph.add_vertex(num_nodes+i)
        for i in range(num_nodes_mules):
            graph.add_edge(num_nodes, num_nodes+i+1) # Connect the source to the money mules
            graph.add_edge(num_nodes+i+1, num_nodes+num_nodes_mules+1) # Connect the money mules to the sink
    
    return graph

def add_smurfing_patterns_new_mules(graph, n):
    for i in range(n):
        # Add smurfing pattern to the graph
        pass

def add_smurfing_patterns_existing_mules(graph, n):
    for i in range(n):
        # Add smurfing pattern to the graph
        pass

def add_smurfing_patterns(graph, n, type=''):
    """
    Add smurfing patterns to the original graph
    :param graph: igraph.Graph object
    :param n: number of smurfing patterns to add
    :param type: type of smurfing pattern to add (separate, new_mules, existing_mules)
    :return: graph_smurfing: igraph.Graph object with smurfing patterns added
    """

    if type == 'separate':
        graph_smurfing = add_smurfing_patterns_separate(graph, n)
    elif type == 'new_mules':
        graph_smurfing = add_smurfing_patterns_new_mules(graph, n)
    elif type == 'existing_mules':
        graph_smurfing = add_smurfing_patterns_existing_mules(graph, n)
    else:
        raise ValueError('Invalid type of smurfing pattern')
    
    return graph_smurfing

if __name__ == '__main__':
    # Create synthetic Barabasi-Albert graph
    graph = ig.Graph()
    ba = graph.Barabasi(2000,m=2)

    ## Add smurfing patterns
    # We implement three types of smurfing patterns: separate, new_mules, existing_mules
    # Separate: Smurfing patterns are separate from the original graph
    # new_mules: Smurfing patterns are constructed using new mules (which only make transactions as money mules)
    # existing_mules: Smurfing patterns are constructed using existing mules (which have made normal transactions in the past)
    # All three patterns are added to study robustness to masking
    graph_smurfing = add_smurfing_patterns(ba, 2, type='separate')