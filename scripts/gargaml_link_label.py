import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.pattern_construction import define_ML_labels

import scienceplots
plt.style.use(['science','notebook'])


def define_gargaml_scores_directed(results_df_measures):
    nodes = []
    gargaml = []
    transposed_gargaml = []
    max_gargaml = []

    for i,line in results_df_measures.iterrows():
        measure_00 = line["measure_00"]
        measure_01 = line["measure_01"]
        measure_02 = line["measure_02"]
        measure_10 = line["measure_10"]
        measure_11 = line["measure_11"]
        measure_12 = line["measure_12"]
        measure_20 = line["measure_20"]
        measure_21 = line["measure_21"]
        measure_22 = line["measure_22"]

        measure_high = np.mean([measure_01, measure_12])
        measure_low = np.mean([measure_10, measure_21, measure_00, measure_02, measure_11, measure_20, measure_22])
        measure = measure_high - measure_low

        measure_high_transpose = np.mean([measure_10, measure_21])
        measure_low_transpose = np.mean([measure_01, measure_12, measure_00, measure_20, measure_11, measure_02, measure_22])
        measure_transpose = measure_high_transpose - measure_low_transpose

        nodes.append(line["node"])
        gargaml.append(measure)
        transposed_gargaml.append(measure_transpose)
        max_gargaml.append(max(measure, measure_transpose))

    dict_gargaml = {
        "node": nodes,
        "GARGAML": gargaml, 
        "GARGAML_transposed": transposed_gargaml,
        "GARGAML_max": max_gargaml
    }

    results_df = pd.DataFrame(dict_gargaml)
    results_df.set_index("node", inplace=True)
    return results_df

def define_gargaml_scores_undirected(results_df_measures):
    nodes = []
    gargaml = []

    for i,line in results_df_measures.iterrows():
        measure_1 = line["measure_1"]
        measure_2 = line["measure_2"]
        measure_3 = line["measure_3"]

        measure = measure_2 - (measure_1 + measure_3)/2

        nodes.append(line["node"])
        gargaml.append(measure)

    dict_gargaml = {
        "node": nodes,
        "GARGAML": gargaml
    }

    results_df = pd.DataFrame(dict_gargaml)
    results_df.set_index("node", inplace=True)
    return results_df

def combine_patterns_GARGAML(results_df, laundering_df, dataset, directed, pattern_columns, name='Combined'):
    gargaml_scores = []
    for account in laundering_df.index:
        try:
            line=results_df.loc[account]
            gargaml_scores.append(line["GARGAML"])
        except:
            gargaml_scores.append(-2)

    laundering_df["GARGAML"] = gargaml_scores

    laundering_df["GARGAML_rounded"] = [round(x, 1) for x in laundering_df["GARGAML"]]


    dir_string = "directed" if directed else "undirected"
    laundering_df[laundering_df["GARGAML"]!=-2].groupby("GARGAML_rounded")[pattern_columns].mean().plot(alpha=0.7, figsize=(10, 7))
    plt.title(dataset+" - "+dir_string+" - "+name)
    plt.savefig("results/"+dataset+"_GARGAML_"+dir_string+"_"+name+".pdf")
    plt.close()

def main():
    dataset = "HI-Small"  
    directed = False

    transactions_df_extended, pattern_columns = define_ML_labels(
        path_trans = "data/"+dataset+"_Trans.csv",
        path_patterns = "data/"+dataset+"_Patterns.txt"
    )

    str_directed = "directed" if directed else "undirected"
    results_df_measures = pd.read_csv("results/"+dataset+"_GARGAML_"+str_directed+".csv")

    if directed:
        results_df = define_gargaml_scores_directed(results_df_measures)
    else:
        results_df = define_gargaml_scores_undirected(results_df_measures)

    laundering_from = transactions_df_extended[["Account", "Is Laundering"]+pattern_columns].groupby("Account").mean()
    laundering_to = transactions_df_extended[["Account.1", "Is Laundering"]+pattern_columns].groupby("Account.1").mean()
    
    trans_from=transactions_df_extended[["Account", "Is Laundering"]+pattern_columns]
    trans_to=transactions_df_extended[["Account.1", "Is Laundering"]+pattern_columns]
    trans_to.columns = ["Account", "Is Laundering"]+pattern_columns
    laundering_combined = pd.concat([trans_from, trans_to]).groupby("Account").mean()

    combine_patterns_GARGAML(results_df, laundering_combined, dataset, directed, pattern_columns, name='Combined')
    combine_patterns_GARGAML(results_df, laundering_from, dataset, directed, pattern_columns, name='Sender')
    combine_patterns_GARGAML(results_df, laundering_to, dataset, directed, pattern_columns, name='Receiver')

main()