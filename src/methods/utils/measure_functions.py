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