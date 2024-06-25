import pandas as pd
import numpy as np

def create_identifiers(df):
    """
    Create a list of identifiers for each row in the dataframe.
    """
    identifyer_list = []

    for i,row in df.iterrows():
        item = ""
        for col in df.columns:
            item += str(row[col]) + ","
        identifyer_list.append(item[:-1])

    return identifyer_list

def create_AML_labels(path= "data/HI-Small_Patterns.txt"):
    transaction_list = []
    fanout_list = []
    fanin_list = []
    gather_scatter_list = []
    scatter_gather_list = []
    cycle_list = []
    random_list = []
    bipartite_list = []
    stack_list = []

    with open(path, "r") as f:
        attemptActive = False
        column = ""

        # Initialize all lists with zeros for simplification
        list_defaults = [0] * 8  # Assuming there are 8 lists as per the code snippet

        # Mapping of column names to their corresponding list index
        column_to_list_index = {
            "FAN-OUT": 0,
            "FAN-IN": 1,
            "GATHER-SCATTER": 2,
            "SCATTER-GATHER": 3,
            "CYCLE": 4,
            "RANDOM": 5,
            "BIPARTITE": 6,
            "STACK": 7
        }
        while True:
            line = f.readline()
            # Check if not at the end of the file
            if not line:
                break

            # Add pattern to the corresponding transaction
            if line.startswith("BEGIN"): # Start of a pattern
                attemptActive = True
                column = line.split(" - ")[1].split(":")[0].strip()
            elif line.startswith("END"): # End of a pattern => reset all parameters + no update of columns
                attemptActive = False
                column = ""
            elif attemptActive:
                identifyer = line.strip()
                transaction_list.append(identifyer)
                
                # Reset all lists to default values
                current_values = list_defaults.copy()
                
                if column in column_to_list_index:
                    # Update the relevant list based on the column name
                    current_values[column_to_list_index[column]] = 1
                    
                    # Unpack the updated values to each list
                    fanout_list.append(current_values[0])
                    fanin_list.append(current_values[1])
                    gather_scatter_list.append(current_values[2])
                    scatter_gather_list.append(current_values[3])
                    cycle_list.append(current_values[4])
                    random_list.append(current_values[5])
                    bipartite_list.append(current_values[6])
                    stack_list.append(current_values[7])

                else:
                    raise ValueError("Unknown pattern type")
                
    df_patterns = pd.DataFrame(
        {
            "Identifyer": transaction_list,
            "FAN-OUT": fanout_list,
            "FAN-IN": fanin_list,
            "GATHER-SCATTER": gather_scatter_list,
            "SCATTER-GATHER": scatter_gather_list,
            "CYCLE": cycle_list,
            "RANDOM": random_list,
            "BIPARTITE": bipartite_list,
            "STACK": stack_list
        }
    )

    return df_patterns

def define_ML_labels(path_trans="data/HI-Small_Trans.csv", path_patterns="data/HI-Small_Patterns.txt"):
    transactions_df = pd.read_csv(path_trans, dtype=str)
    transactions_df['Is Laundering'] = transactions_df['Is Laundering'].astype(int)
    
    identifyer_list = create_identifiers(transactions_df)
    transactions_df["Identifyer"] = identifyer_list
    del identifyer_list

    pattern_columns = ["FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER", "CYCLE", "RANDOM", "BIPARTITE", "STACK"]
    df_patterns = create_AML_labels(path_patterns)

    # Merge the two dataframes
    transactions_df_extended = transactions_df.merge(df_patterns, on="Identifyer", how="left")
    transactions_df_extended = transactions_df_extended.fillna(0)

    # Vectorized check for "Is Laundering" being 1
    is_laundering = transactions_df_extended["Is Laundering"] == 1

    # Vectorized sum of specified columns
    pattern_sum = transactions_df_extended[pattern_columns].sum(axis=1)

    # Use numpy.where for a vectorized conditional operation
    transactions_df_extended["Not Classified"] = np.where((is_laundering) & (pattern_sum == 0), 1, 0)
        
    pattern_columns.append("Not Classified")

    return transactions_df_extended, pattern_columns