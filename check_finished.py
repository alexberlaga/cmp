import os
import sys
import pandas as pd
import numpy as np
import itertools

def get_peptoids_from_round(i):
    """
    This function reads the "bo_peptoids.csv" file, and extracts the peptoids for the specified round.

    Args:
        i (int): The round number.

    Returns:
        list: A list of peptoids for the specified round.
    """
    all_peps = pd.read_csv('bo_peptoids.csv', index_col=0)
    round = all_peps.iloc[i].values.flatten().tolist()
    round = [s for s in round if isinstance(s, str)]
    round = [s for s in round if s[-1] == 'G']
    return round

cd = os.getcwd()


round = int(sys.argv[1])
peps = get_peptoids_from_round(round)
os.chdir('small_melting_temps')

for folder in peps:
    if os.path.isdir(folder) and folder[-1] == 'G':
        counter = 0
        for j in range(6):
            child_folder = os.path.join(folder, '2_anneal')
            if os.path.isfile(os.path.join(child_folder, str(j)+ '_annealing_output.txt')):
                temps = np.loadtxt(os.path.join(child_folder, str(j)+'_annealing_output.txt'))
                if temps[-1, 3] <= 0.8:
                    counter += 1
                    
        print(folder, counter)

os.chdir(cd)
