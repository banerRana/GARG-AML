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
            num_nodes_mules = random.randint(2, 10) # Randomly have two to five money mules

        for j in range(num_nodes_mules+2): # Add two extra nodes for the source and target
            graph.add_vertex(num_nodes+j)
            graph.vs[num_nodes+j]['new_node'] = True
            graph.vs[num_nodes+j]['laundering'] = True
            graph.vs[num_nodes+j]['separate'] = True
        for k in range(num_nodes_mules):
            graph.add_edge(num_nodes, num_nodes+k+1) # Connect the source to the money mules
            graph.es[graph.get_eid(num_nodes, num_nodes+k+1)]['laundering'] = True
            graph.add_edge(num_nodes+k+1, num_nodes+num_nodes_mules+1) # Connect the money mules to the target
            graph.es[graph.get_eid(num_nodes+k+1, num_nodes+num_nodes_mules+1)]['laundering'] = True
    
    return graph

def add_smurfing_patterns_new_mules(graph, n, num_smurfs):
    for i in range(n):
        # Add smurfing pattern to the graph

        # Determine number of nodes in the graph
        num_nodes = graph.vcount()

        # Determine number of nodes in the smurfing pattern
        if len(num_smurfs) == 0:
            num_nodes_mules = random.randint(2, 10) # Randomly have two to five money mules
        else:
            num_nodes_mules = num_smurfs[i]
        
        # Randomly select source and target node from existing nodes
        source_node = list_nodes.pop()
        target_node = list_nodes.pop()
        graph.vs[source_node]['laundering'] = True
        graph.vs[target_node]['laundering'] = True
        graph.vs[source_node]['new_mules'] = True
        graph.vs[target_node]['new_mules'] = True

        for j in range(num_nodes_mules):
            graph.add_vertex(num_nodes+j)
            graph.vs[num_nodes+j]['new_node'] = True
            graph.vs[num_nodes+j]['laundering'] = True
            graph.vs[num_nodes+j]['new_mules'] = True
        for k in range(num_nodes_mules):
            graph.add_edge(source_node, num_nodes+k) # Connect the source to the money mules
            graph.es[graph.get_eid(source_node, num_nodes+k)]['laundering'] = True
            graph.add_edge(num_nodes+k, target_node) # Connect the money mules to the target
            graph.es[graph.get_eid(num_nodes+k, target_node)]['laundering'] = True
    
    return graph


def add_smurfing_patterns_existing_mules(graph, n, num_smurfs):
    for i in range(n):
        # Add smurfing pattern to the graph

        # Determine number of nodes in the smurfing pattern
        if len(num_smurfs) == 0:
            num_nodes_mules = random.randint(2, 10) # Randomly have two to five money mules
        else:
            num_nodes_mules = num_smurfs[i]
    
        # Randomly select source and target node from existing nodes
        source_node = list_nodes.pop()
        target_node = list_nodes.pop()
        graph.vs[source_node]['laundering'] = True
        graph.vs[target_node]['laundering'] = True
        graph.vs[source_node]['existing_mules'] = True
        graph.vs[target_node]['existing_mules'] = True

        for i in range(num_nodes_mules):
            node = list_nodes.pop()
            graph.vs[node]['laundering'] = True
            graph.vs[node]['existing_mules'] = True
            graph.add_edge(source_node, node)
            graph.es[graph.get_eid(source_node, node)]['laundering'] = True
            graph.add_edge(node, target_node)
            graph.es[graph.get_eid(node, target_node)]['laundering'] = True

    return graph

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
    ba = graph.Barabasi(200, m=1)

    list_nodes = ba.vs.indices
    random.shuffle(list_nodes)

    ba.vs['new_node'] = False 
    ba.vs['laundering'] = False

    ba.vs['separate'] = False
    ba.vs['new_mules'] = False
    ba.vs['existing_mules'] = False

    ba.es['laundering'] = False

    ## Add smurfing patterns
    # We implement three types of smurfing patterns: separate, new_mules, existing_mules
    # Separate: Smurfing patterns are separate from the original graph
    # new_mules: Smurfing patterns are constructed using new mules (which only make transactions as money mules)
    # existing_mules: Smurfing patterns are constructed using existing mules (which have made normal transactions in the past)
    # All three patterns are added to study robustness to masking
    graph_smurfing = add_smurfing_patterns(ba, 3, type_pattern='separate')
    graph_smurfing = add_smurfing_patterns(ba, 3, type_pattern='new_mules')
    graph_smurfing = add_smurfing_patterns(ba, 3, type_pattern='existing_mules')
    graph_smurfing.simplify(combine_edges='max')

    graph_smurfing.vs['color'] = ['red' if x['new_node'] else 'blue' for x in graph_smurfing.vs]
    graph_smurfing.vs['size'] = [10 if x['laundering'] else 5 for x in graph_smurfing.vs]
    graph_smurfing.es['color'] = ['black' if x['laundering'] else 'gray' for x in graph_smurfing.es]

    visual_style = {}
    visual_style["vertex_size"] = graph_smurfing.vs['size']
    visual_style["vertex_color"] = graph_smurfing.vs['color']
    visual_style["edge_color"] = graph_smurfing.es['color']
    visual_style["edge_width"] = 1
    ig.plot(graph_smurfing, **visual_style, target="data/visualisation_network.pdf")

    # Extract nodes into a pandas dataframe
    nodes_data = {attr: graph_smurfing.vs[attr] for attr in graph_smurfing.vs.attributes()}
    nodes_df = pd.DataFrame(nodes_data)[['separate', 'new_mules', 'existing_mules', 'laundering']]
    nodes_df.fillna(False, inplace=True)

    # Extract edges into a pandas dataframe
    source_list = []
    target_list = []
    for edge in graph_smurfing.es:
        source = edge.source
        target = edge.target
        source_list.append(source)
        target_list.append(target)
    
    edges_df = pd.DataFrame({'source': source_list, 'target': target_list})

    print('Number of nodes:', graph_smurfing.vcount())
    print('Number of edges:', graph_smurfing.ecount())

    # Save the dataframe to a CSV file
    nodes_df.to_csv('data/label_data_synthetic.csv', index=False)
    edges_df.to_csv('data/edge_data_synthetic.csv', index=False)