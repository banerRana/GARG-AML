import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

from src.data.pattern_construction import define_ML_labels, summarise_ML_labels
from src.methods.gargaml_scores import define_gargaml_scores
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

def divergence_metric(dist_0, dist_1):
    mean_0 = np.mean(dist_0)
    variance_0 = np.var(dist_0)
    mean_1 = np.mean(dist_1)
    variance_1 = np.var(dist_1)
    return (mean_0 - mean_1)**2 + 0.5*(variance_0 + variance_1)

def lift_curve_values(y_val, y_pred, steps):
    vals_lift = [] #The lift values to be plotted

    df_lift = pd.DataFrame()
    df_lift['Real'] = y_val
    df_lift['Pred'] = y_pred
    df_lift.sort_values('Pred',
                        ascending=False,
                        inplace=True)

    global_ratio = df_lift['Real'].sum() / len(df_lift['Real'])

    for step in steps:
        data_len = int(np.ceil(step*len(df_lift)))
        data_lift = df_lift.iloc[:data_len, :]
        val_lift = data_lift['Real'].sum()/data_len
        vals_lift.append(val_lift/global_ratio)

    return(vals_lift)

dataset = "HI-Small"  
directed = False
score_type = "basic" # basic or weighted_average

str_directed = "directed" if directed else "undirected"
results_df_measures = pd.read_csv("results/"+dataset+"_GARGAML_"+str_directed+".csv")

results_df = define_gargaml_scores(results_df_measures, directed=directed, score_type=score_type)

transactions_df_extended, pattern_columns = define_ML_labels(
    path_trans = "data/"+dataset+"_Trans.csv",
    path_patterns = "data/"+dataset+"_Patterns.txt"
)

laundering_combined, _, _ = summarise_ML_labels(transactions_df_extended,pattern_columns)

from_data = transactions_df_extended[["Account", "From Bank"]].drop_duplicates()
from_data.columns = ["Account", "Bank"]
to_data = transactions_df_extended[["Account.1", "To Bank"]].drop_duplicates()
to_data.columns = ["Account", "Bank"]
total_data = pd.concat([from_data, to_data], axis=0).drop_duplicates()

cut_offs = [0.1, 0.2, 0.3, 0.5, 0.9]
columns = ['Is Laundering', 'FAN-OUT', 'FAN-IN', 'GATHER-SCATTER', 'SCATTER-GATHER', 'CYCLE', 'RANDOM', 'BIPARTITE', 'STACK']

n = len(cut_offs)
m = len(columns)

divergence_matrix = np.zeros((n, m))

fig, axes = plt.subplots(n, m, figsize = (n*5, m))

for i in range(n):
    cut_off = cut_offs[i]
    for j in range(m):
        column = columns[j]

        laundering_combined["Label"] = ((laundering_combined[column]>cut_off)*1).values

        labels = []
        for node in results_df.index:
            label = int(laundering_combined.loc[node]["Label"])
            labels.append(label)

        results_df["Label"] = labels
    
        # Filter the DataFrame by label
        label_0 = results_df[results_df["Label"] == 0]["GARGAML"]
        label_1 = results_df[results_df["Label"] == 1]["GARGAML"]

        # Calculate the bin edges
        all_data = np.concatenate([label_0, label_1])
        bins = np.histogram_bin_edges(all_data, bins=20)

        # Plot histogram for label 0
        axes[i, j].hist(label_0, bins=bins, alpha=0.5, label='Label 0', density=True)

        # Plot histogram for label 1
        axes[i, j].hist(label_1, bins=bins, alpha=0.5, label='Label 1', density=True)

        # Add labels and title
        #axes[i, j].set_xlabel('GARGAML')
        #axes[i, j].set_ylabel('Relative Frequency')
        #axes[i, j].set_title('Label: "'+ column +'" at '+ str(cut_off))
        #axes[i, j].legend()

        divergence = divergence_metric(label_0, label_1)
        divergence_matrix[i, j] = divergence

divergence_df = pd.DataFrame(divergence_matrix, columns=columns, index=cut_offs)
divergence_df.to_csv("results/"+dataset+"_GARGAML_"+str_directed+"_combined_divergence.csv")

for axis, col in zip(axes[0], columns):
    axis.set_title(col)

for axis, row in zip(axes[:,0], cut_offs):
    axis.set_ylabel(row, size='large')

fig.suptitle('Distribution of '+ str_directed +' GARGAML Scores by Label for data set: '+ dataset)
fig.tight_layout()

plt.savefig("results/"+dataset+"_GARGAML_"+str_directed+"_combined_histogram.pdf")

fig, axes = plt.subplots(n, m, figsize = (n*5, m))
values = np.linspace(0.01, 1, 100)

for i in range(n):
    cut_off = cut_offs[i]
    for j in range(m):
        column = columns[j]

        laundering_combined["Label"] = ((laundering_combined[column]>cut_off)*1).values

        labels = []
        for node in results_df.index:
            label = int(laundering_combined.loc[node]["Label"])
            labels.append(label)

        results_df["Label"] = labels
    
        lift = lift_curve_values(results_df["Label"], results_df["GARGAML"], values)

        axes[i, j].plot(values, lift)

for axis, col in zip(axes[0], columns):
    axis.set_title(col)

for axis, row in zip(axes[:,0], cut_offs):
    axis.set_ylabel(row, size='large')

fig.suptitle('Lift curve of '+ str_directed +' GARGAML Scores by Label for data set: '+ dataset)
fig.tight_layout()

plt.savefig("results/"+dataset+"_GARGAML_"+str_directed+"_combined_lift.pdf")