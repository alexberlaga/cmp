import sys, os
sys.path.append("/project2/andrewferguson/berlaga/peptoids/sams")
import numpy as np
import torch
import pickle
import pandas as pd
import gc
from scipy.stats import sem
from encoding import ALL_PEPS

# Path to the SAMS data directory
TM_PATH = "/project2/andrewferguson/berlaga/activelearning/melting_temps"

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
    if np.any(bool_array1) and np.all(bool_array2):
        return np.argwhere(bool_array1).flatten()[-1] + 1
    else:
        return np.inf

def get_current_round(prev=False):
    """
    This function reads the "bo_energies.csv" file and finds the current round based on the presence of NaN values in the "e1" column.

    Args:
        prev (bool, optional): Whether to get the previous round instead of the current round. Defaults to False.

    Returns:
        int: The current round number.
    """
    all_energies = pd.read_csv('bo_tm.csv', index_col=0)
    e1 = all_energies['t1'].to_numpy()
    if prev:
        return np.min(np.where(np.isnan(e1))) - 1
    return np.min(np.where(np.isnan(e1)))

ROUND = get_current_round()    


def update_round():
    global ROUND
    ROUND = get_current_round()    



def get_idxs_up_to(round=ROUND):
    return np.concatenate([[ALL_PEPS.index(p) for p in get_peptoids_from_round(r)] for r in range(round+1)])

def get_peptoids_up_to(round=ROUND):
    return np.concatenate([get_peptoids_from_round(r) for r in range(round+1)])

def get_tm_up_to(round=ROUND):
    return torch.Tensor(np.concatenate([get_tm_from_round(r) for r in range(round+1)]))

def get_noise_up_to(round=ROUND):
    return torch.Tensor(np.concatenate([get_noise_from_round(r) for r in range(round+1)]))

    
def extract_tm(peptoid_string, noise=True):
    """
    Extracts the probability a peptoid is in the PPII region.

    Args:
        peptoid_string (str): The name of the peptoid (e.g., "PPGPPG").

    Returns:
        float: The extracted probability.
    """
    
    
    temp_folder = os.path.join(TM_PATH, peptoid_string, "3_anneal")
    cur_temps = np.zeros(6)
    for j in range(6):
        colvar = np.loadtxt(os.path.join(temp_folder, str(j)+'_annealing_output.txt'))
        first_big_idx = np.min((get_first_big_idx(colvar[-100:, 4]), get_first_big_idx(colvar[-100:, 5]), get_first_big_idx(colvar[-100:, 6])))
        first_small_idx = np.min((get_first_small_idx(colvar[-100:, 1]), get_first_small_idx(colvar[-100:, 2]), get_first_small_idx(colvar[-100:, 3])))
        first_idx = min(first_big_idx, first_small_idx)
        if not np.isinf(first_idx):
            cur_temps[j] = min(320 + colvar[-100 + int(first_idx), 0] * 0.001, 520)
   
    correct_temps = cur_temps[cur_temps != 0]
    if noise:
        return np.mean(correct_temps), sem(correct_temps)
    return np.mean(correct_temps)



def all_tm_files():
    """
    Finds all valid SAMS files for analysis.

    Returns:
        list: A list of valid SAMS file names.
    """
    files = []
    for file in os.listdir(TM_PATH):
        if len(file) == 6 and file[2] == 'G' and file[5] == 'G' and file[0] != 'G':
            if os.path.isfile(os.path.join(TM_PATH, file, "2_sams", "ee_whole.pdb")):
                files.append(file)
    return files



def extract_tm_from_round(round):
    """
    Extracts energies for peptoids in a specific round from a CSV file.

    Args:
        i (int): The round number.

    Returns:
        np.ndarray: An array of energies for the peptoids in the round.
    """

    peps = get_peptoids_from_round(round)
    tm_means = list(map(lambda p: extract_tm(p, noise=True)[0], peps))
    tm_stds = list(map(lambda p: extract_tm(p, noise=True)[1], peps))
    return np.array(tm_means), np.array(tm_stds)



def save_peptoids(peptoids, i):
    """
    Saves a list of peptoids to a CSV file for a specific round.

    Args:
        peptoids (list): A list of peptoid names
        i (int): The round number.
    """
    all_peps = pd.read_csv("bo_peptoids.csv", index_col=0)
    for n, pep in enumerate(peptoids):
        all_peps.loc[:, "p" + str(n+1)].iloc[i] = pep

    all_peps.to_csv("bo_peptoids.csv")
    
def save_tm_and_peptoids(peptoids, i):
    """
    Saves both peptoids and their corresponding energies to CSV files for a specific round.

    Args:
        peptoids (list): A list of peptoid names.
        i (int): The round number.
    """

    energies = np.zeros(len(peptoids))
    for n, pep in enumerate(peptoids):
        energies[n] = extract_tm(pep)
    save_peptoids(peptoids, i)
    save_tm(energies, i)

def save_tm(energies, i):
    """
    Saves a list of energies to a CSV file for a specific round.

    Args:
        energies (list): A list of stability energy values.
        i (int): The round number.
    """
    all_energies = pd.read_csv("bo_tm.csv", index_col=0)
    for n, e in enumerate(energies):
        all_energies.loc[:, "t" + str(n+1)].iloc[i] = e
    all_energies.to_csv("bo_tm.csv")

def save_noise(noise, i):
    """
    Saves a list of energies to a CSV file for a specific round.

    Args:
        energies (list): A list of stability energy values.
        i (int): The round number.
    """
    all_errs = pd.read_csv("bo_noise.csv", index_col=0)
    for n, e in enumerate(noise):
        all_errs.loc[:, "e" + str(n+1)].iloc[i] = e
    all_errs.to_csv("bo_noise.csv")

def load_model(round, filename='gprs/round'):
    """
    This function loads a pickled optimization model from a file.

    Args:
        round (int): The round number.
        filename (str, optional): The base filename of the model file. Defaults to 'optimizers/optimizer'.

    Returns:
        object: The loaded optimization model.
    """

    openfile = filename + "_" + str(round) + ".pt"
    return torch.load(openfile)


def save_model(model, round, filename='gprs/round'):
    """
    This function saves a pickled optimization model to a file.

    Args:
        optimizer (object): The optimization model to save.
        round (int): The round number.
        filename (str, optional): The base filename of the model file. Defaults to 'optimizers/optimizer'.
    """
    openfile = filename + "_" + str(round) + ".pt"
    torch.save(model, openfile)
        
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

def get_tm_from_round(i):
    """
    This function reads the "bo_energies.csv" file, extracts the energies for the specified round, and filters out any NaN values.

    Args:
        i (int): The round number.

    Returns:
        list: A list of energies for the specified round.
    """
    all_energies = pd.read_csv('bo_tm.csv', index_col=0)
    round = all_energies.iloc[i].values.flatten()[:]
    round = round[np.logical_not(np.isnan(round))]
    return round

def get_noise_from_round(i):
    """
    This function reads the "bo_energies.csv" file, extracts the energies for the specified round, and filters out any NaN values.

    Args:
        i (int): The round number.

    Returns:
        list: A list of energies for the specified round.
    """
    all_energies = pd.read_csv('bo_noise.csv', index_col=0)
    round = all_energies.iloc[i].values.flatten()[:]
    round = round[np.logical_not(np.isnan(round))]
    return round


def extract_known_tm(peptoid):
    prev_peps = list(np.concatenate([[p for p in get_peptoids_from_round(r)] for r in range(ROUND)]))
    prev_energies = list(np.concatenate([[p for p in get_tm_from_round(r)] for r in range(ROUND)]))
    return prev_energies[prev_peps.index(peptoid)]

def record_score(round, sc):
    """
    This function loads the scores from a file, updates the score for the current round if specified, and saves the scores back to the file. 

    Args:
        round (int): The current round.
        sc (float): The score for the current round.
    """
    scores = np.load('scores.npy')
    scores[round] = sc
    np.save('scores.npy', scores)

def record_maxEI(ei, round):
    """
    This function loads the scores from a file, updates the score for the current round if specified, and saves the scores back to the file. 

    Args:
        round (int): The current round.
        ei (float): The max EI for the current round.
    """
    eis = np.load('max_eis.npy')
    eis[round] = ei
    np.save('max_eis.npy', eis)
