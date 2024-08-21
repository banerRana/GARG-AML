import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import pandas as pd
from tqdm import tqdm

from src.data.graph_construction import construct_IBM_graph
from src.utils.graph_processing import graph_community
from src.methods.GARGAML import GARG_AML_node_undirected_measures

dataset = "HI-Small"
path = "data/"+dataset+"_Trans.csv"
directed = False

G = construct_IBM_graph(path=path, directed = directed)
G_reduced = graph_community(G)
measure_1_list = []
measure_2_list = []
measure_3_list = []
size_1_list = []
size_2_list = []
size_3_list = []

nodes = list(G_reduced.nodes)

for node in tqdm(nodes):
    measure_1, measure_2, measure_3, size_1, size_2, size_3 = GARG_AML_node_undirected_measures(node, G_reduced, include_size=True)
    measure_1_list.append(measure_1)
    measure_2_list.append(measure_2)
    measure_3_list.append(measure_3)
    size_1_list.append(size_1)
    size_2_list.append(size_2)
    size_3_list.append(size_3)

data_dict = {
    "node": nodes,
    "measure_1": measure_1_list,
    "measure_2": measure_2_list,
    "measure_3": measure_3_list,
    "size_1": size_1_list,
    "size_2": size_2_list,
    "size_3": size_3_list
}

df = pd.DataFrame(data_dict)
df.to_csv("results/"+dataset+"_GARGAML_undirected.csv", index=False)
