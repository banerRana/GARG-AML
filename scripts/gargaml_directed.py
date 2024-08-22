import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import pandas as pd

from src.data.graph_construction import construct_IBM_graph
from src.utils.graph_processing import graph_community
from src.methods.GARGAML import GARG_AML_node_directed_measures

dataset = "HI-Small"
path = "data/"+dataset+"_Trans.csv"
directed = True

G = construct_IBM_graph(path=path, directed = directed)
G_reduced = graph_community(G)

G_reduced_und = G_reduced.to_undirected()
G_reduced_rev = G_reduced.reverse(copy=True)

measure_00_list = []
measure_01_list = []
measure_02_list = []
measure_10_list = []
measure_11_list = []
measure_12_list = []
measure_20_list = []
measure_21_list = []
measure_22_list = []

size_00_list = []
size_01_list = []
size_02_list = []
size_10_list = []
size_11_list = []
size_12_list = []
size_20_list = []
size_21_list = []
size_22_list = []

nodes = list(G_reduced.nodes)

for node in nodes:
    (
        measure_00, measure_01, measure_02, measure_10, measure_11, measure_12, measure_20, measure_21, measure_22, 
        size_00, size_01, size_02, size_10, size_11, size_12, size_20, size_21, size_22 
    ) = GARG_AML_node_directed_measures(node, G_reduced, G_reduced_und, G_reduced_rev, include_size=True)
    
    measure_00_list.append(measure_00)
    measure_01_list.append(measure_01)
    measure_02_list.append(measure_02)
    measure_10_list.append(measure_10)
    measure_11_list.append(measure_11)
    measure_12_list.append(measure_12)
    measure_20_list.append(measure_20)
    measure_21_list.append(measure_21)
    measure_22_list.append(measure_22)

    size_00_list.append(size_00)
    size_01_list.append(size_01)
    size_02_list.append(size_02)
    size_10_list.append(size_10)
    size_11_list.append(size_11)
    size_12_list.append(size_12)
    size_20_list.append(size_20)
    size_21_list.append(size_21)
    size_22_list.append(size_22)

data_dict = {
    "node": nodes,
    "measure_00": measure_00_list,
    "measure_01": measure_01_list,
    "measure_02": measure_02_list,
    "measure_10": measure_10_list,
    "measure_11": measure_11_list,
    "measure_12": measure_12_list,
    "measure_20": measure_20_list,
    "measure_21": measure_21_list,
    "measure_22": measure_22_list, 
    "size_00": size_00_list,
    "size_01": size_01_list,
    "size_02": size_02_list,
    "size_10": size_10_list,
    "size_11": size_11_list,
    "size_12": size_12_list,
    "size_20": size_20_list,
    "size_21": size_21_list,
    "size_22": size_22_list
}

df = pd.DataFrame(data_dict)
df.to_csv("results/"+dataset+"_GARGAML_directed.csv", index=False)