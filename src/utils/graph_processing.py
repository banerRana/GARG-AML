import pandas as pd
import networkx as nx
import numpy as np

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