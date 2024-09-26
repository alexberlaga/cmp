import os
import sys
import pandas as pd
import numpy as np
import itertools



DIST_THRESH = 2
SIZE_THRESH = 1.25

def get_first_big_idx(a, thresh=DIST_THRESH):
    bool_array = a > thresh
    if np.any(bool_array):
        return np.argmax(bool_array)
    else:
        return np.inf

def get_first_small_idx(a, thresh=SIZE_THRESH, num_small=10):
    bool_array1 = a > thresh
    bool_array2 = a[-num_small:] < thresh
    if not np.any(bool_array1):
        return num_small
    if np.any(bool_array1) and np.all(bool_array2):
        return np.argwhere(bool_array1).flatten()[-1] + 1
    else:
        return np.inf
        
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
os.chdir('melting_temps')

for folder in peps:
    if os.path.isdir(folder) and folder[-1] == 'G':
        counter = 0
        child_folder = os.path.join(folder, '3_anneal')
        if os.path.isfile(os.path.join(child_folder, 'error.out')):
            error_string = open(os.path.join(child_folder, 'error.out'), "r").read()
            if "MPI_ABORT" in error_string:
                print(folder, "MPI_ABORT")
        
        for j in range(6):
            if os.path.isfile(os.path.join(child_folder, str(j)+ '_annealing_output.txt')):
                temps = np.loadtxt(os.path.join(child_folder, str(j)+'_annealing_output.txt'))
                colvar = np.loadtxt(os.path.join(child_folder, str(j)+'_annealing_output.txt'))
                first_big_idx = np.min((get_first_big_idx(colvar[-100:, 4]), get_first_big_idx(colvar[-100:, 5]), get_first_big_idx(colvar[-100:, 6])))
                first_small_idx = np.min((get_first_small_idx(colvar[-100:, 1]), get_first_small_idx(colvar[-100:, 2]), get_first_small_idx(colvar[-100:, 3])))
                first_idx = min(first_big_idx, first_small_idx)
               	if not np.isinf(first_idx):
                    counter += 1
        print(folder, counter)
    else:
        print("no sim:", folder)
os.chdir(cd)
