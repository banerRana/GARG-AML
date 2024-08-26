from sklearn import tree

import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.pattern_construction import define_ML_labels, summarise_ML_labels, combine_patterns_GARGAML
from src.methods.gargaml_scores import define_gargaml_scores, summarise_gargaml_scores
from src.data.graph_construction import construct_IBM_graph
from src.utils.graph_processing import graph_community

from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from pickle import dump

def gargaml_tree(X, y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)

    with open("results/model_tree.pkl", "wb") as f:
        dump(clf, f, protocol=5)

    return clf

def gargaml_boosting(X, y):
    clf = ensemble.GradientBoostingClassifier()
    clf = clf.fit(X, y)

    with open("results/model_xgb.pkl", "wb") as f:
        dump(clf, f, protocol=5)

    return clf

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    AUC_ROC = roc_auc_score(y_test, y_pred)
    AUC_PR = average_precision_score(y_test, y_pred)
    return AUC_ROC, AUC_PR

def main():
    dataset = "HI-Small"  
    directed = True
    score_type = "basic"
    target = "SCATTER-GATHER"
    cutoff = 0.2

    gargaml_columns = [
        "GARGAML", 
        "GARGAML_min", "GARGAML_max", "GARGAML_mean", "GARGAML_std",
        "degree", "degree_min", "degree_max", "degree_mean", "degree_std"
        ]

    str_directed = "directed" if directed else "undirected"
    results_df_measures = pd.read_csv("results/"+dataset+"_GARGAML_"+str_directed+".csv") #measures

    results_df = define_gargaml_scores(results_df_measures, directed, score_type=score_type) #summary scores

    transactions_df_extended, pattern_columns = define_ML_labels( #patterns
        path_trans = "data/"+dataset+"_Trans.csv",
        path_patterns = "data/"+dataset+"_Patterns.txt"
    )

    path = "data/"+dataset+"_Trans.csv"
    G = construct_IBM_graph(path=path, directed = False) #For summary, we use the undirected graph
    G_reduced = graph_community(G)

    summary_gargaml = summarise_gargaml_scores(G_reduced, results_df, columns = gargaml_columns)
    for column in gargaml_columns:
        results_df[column] = summary_gargaml[column]

    laundering_combined, _, _ = summarise_ML_labels(transactions_df_extended,pattern_columns)

    combined_patterns_GARGAML = combine_patterns_GARGAML(results_df, laundering_combined, columns = gargaml_columns)

    for column in gargaml_columns:
        laundering_combined[column] = combined_patterns_GARGAML[column]

    X = laundering_combined[gargaml_columns]
    rel_labels = laundering_combined[target]
    y = (rel_labels>cutoff)*1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1997, stratify=y)

    gargaml_clf = gargaml_tree(X_train, y_train)

    AUC_ROC, AUC_PR = evaluate_model(gargaml_clf, X_test, y_test)


    print("AUC ROC: ", AUC_ROC)
    print("AUC PR: ", AUC_PR)

if __name__ == "__main__":
    main()