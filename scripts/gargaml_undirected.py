import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import pandas as pd

from src.data.graph_construction import construct_IBM_graph
from src.utils.graph_processing import graph_community
from src.methods.GARGAML import GARG_AML_node_undirected_measures

dataset = "LI-Large"
path = "data/"+dataset+"_Trans.csv"
directed = False

G = construct_IBM_graph(path=path, directed = directed)
G_reduced = graph_community(G, 10)
measure_1_list = []
measure_2_list = []
measure_3_list = []

nodes = list(G_reduced.nodes)

for node in nodes:
    measure_1, measure_2, measure_3 = GARG_AML_node_undirected_measures(node, G_reduced)
    measure_1_list.append(measure_1)
    measure_2_list.append(measure_2)
    measure_3_list.append(measure_3)

data_dict = {
    "node": nodes,
    "measure_1": measure_1_list,
    "measure_2": measure_2_list,
    "measure_3": measure_3_list
}

df = pd.DataFrame(data_dict)
df.to_csv("results/"+dataset+"_GARGAML_undirected.csv", index=False)
