import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import random

def add_smurfing_patterns_separate(graph, n, num_smurfs):
    for i in range(n):
        # Add smurfing pattern to the graph

        # Determine number of nodes in the graph
        num_nodes = graph.vcount()

        # Determine number of nodes in the smurfing pattern
        try:
            num_nodes_mules = num_smurfs[i]
        except:
            num_nodes_mules = random.randint(2, 5) # Randomly have two to five money mules

        for i in range(num_nodes_mules+2):
            graph.add_vertex(num_nodes+i)
        for i in range(num_nodes_mules):
            graph.add_edge(num_nodes, num_nodes+i+1) # Connect the source to the money mules
            graph.add_edge(num_nodes+i+1, num_nodes+num_nodes_mules+1) # Connect the money mules to the target
    
    return graph

def add_smurfing_patterns_new_mules(graph, n, num_smurfs):
    for i in range(n):
        # Add smurfing pattern to the graph

        # Determine number of nodes in the graph
        num_nodes = graph.vcount()

        # Determine number of nodes in the smurfing pattern
        try:
            num_nodes_mules = num_smurfs[i]
        except:
            num_nodes_mules = random.randint(2, 5) # Randomly have two to five money mules
        
        # Randomly select source and target node from existing nodes
        node_list = graph.vs.indices
        random.shuffle(node_list)
        source_node = node_list[0]
        target_node = node_list[1]

        for i in range(num_nodes_mules):
            graph.add_vertex(num_nodes+i)
        for i in range(num_nodes_mules):
            graph.add_edge(source_node, num_nodes+i) # Connect the source to the money mules
            graph.add_edge(num_nodes+i, target_node) # Connect the money mules to the target
    
    return graph


def add_smurfing_patterns_existing_mules(graph, n, num_smurfs):
    for i in range(n):
        # Add smurfing pattern to the graph
        pass

def add_smurfing_patterns(graph, n, num_smurfs=[] ,type_pattern=''):
    """
    Add smurfing patterns to the original graph
    :param graph: igraph.Graph object
    :param n: number of smurfing patterns to add
    :param num_smurfs: number of smurfs to add to each smurfing pattern
    :param type: type of smurfing pattern to add (separate, new_mules, existing_mules)
    :return: graph_smurfing: igraph.Graph object with smurfing patterns added
    """

    assert type(num_smurfs) == list, 'Number of smurfs should be a list'
    assert len(num_smurfs) in [0, 1, n], 'Number of smurfs should be either empty, one value or n values, with n number of smurfing patterns'

    if len(num_smurfs) == 1:
        num_smurfs = num_smurfs * n

    if type_pattern == 'separate':
        graph_smurfing = add_smurfing_patterns_separate(graph, n, num_smurfs)
    elif type_pattern == 'new_mules':
        graph_smurfing = add_smurfing_patterns_new_mules(graph, n, num_smurfs)
    elif type_pattern == 'existing_mules':
        graph_smurfing = add_smurfing_patterns_existing_mules(graph, n, num_smurfs)
    else:
        raise ValueError('Invalid type of smurfing pattern')
    
    return graph_smurfing

if __name__ == '__main__':
    # Create synthetic Barabasi-Albert graph
    graph = ig.Graph()
    ba = graph.Barabasi(20,m=2)

    ## Add smurfing patterns
    # We implement three types of smurfing patterns: separate, new_mules, existing_mules
    # Separate: Smurfing patterns are separate from the original graph
    # new_mules: Smurfing patterns are constructed using new mules (which only make transactions as money mules)
    # existing_mules: Smurfing patterns are constructed using existing mules (which have made normal transactions in the past)
    # All three patterns are added to study robustness to masking
    graph_smurfing = add_smurfing_patterns(ba, 2, type_pattern='new_mules')
    ig.plot(graph_smurfing, target="myfile.pdf")