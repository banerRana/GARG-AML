import pandas as pd
import igraph as ig

def add_smurfing_patterns_separate(graph, n):
    for i in range(n):
        # Add smurfing pattern to the graph
        pass

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

if '__name__' == '__main__':
    # Create synthetic Barabasi-Albert graph
    graph = ig.Graph()
    ba = graph.Barabasi(2000,m=2)

    ## Add smurfing patterns
    # We implement three types of smurfing patterns: separate, new_mules, existing_mules
    # Separate: Smurfing patterns are separate from the original graph
    # new_mules: Smurfing patterns are constructed using new mules (which only make transactions as money mules)
    # existing_mules: Smurfing patterns are constructed using existing mules (which have made normal transactions in the past)
    graph_smurfing = add_smurfing_patterns(graph, 5, type='separate')



