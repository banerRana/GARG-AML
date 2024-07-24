# LOAD MODULES
# Standard library
import os
import sys
DIR = "../"
os.chdir(DIR)
sys.path.append(DIR)

import pandas as pd

def main(split=False):
    if split:
        full_data_set = pd.read_csv("data/LI-Large_Trans.csv")

        n = int(len(full_data_set)/5) # We devide the dataset in 5 parts
        for i in range(5):
            if i == 4:
                full_data_set.iloc[i*n:].to_csv("data/LI-Large_Trans_"+str(i)+".csv", index=False)
            else:
                full_data_set.iloc[i*n:(i+1)*n].to_csv("data/LI-Large_Trans_"+str(i)+".csv", index=False)
    
    else:
        full_data_set = pd.DataFrame()
        for i in range(5):
            df_piece = pd.read_csv("data/LI-Large_Trans_"+str(i)+".csv")
            full_data_set = pd.concat([full_data_set, df_piece])
        full_data_set.reset_index(drop=True, inplace=True)

        full_data_set.to_csv("data/LI-Large_Trans_vsc.csv", index=False)

main()
