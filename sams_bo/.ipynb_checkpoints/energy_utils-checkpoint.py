import sys, os
sys.path.append("/project2/andrewferguson/berlaga/peptoids/sams")
import numpy as np
import mdtraj as md
from sklearn.decomposition import PCA
import pickle
from utils import compute_all_distances
import pandas as pd
import gc
# Path to the SAMS data directory
SAMS_PATH = "/project2/andrewferguson/berlaga/peptoids/sams"

# Load the pre-computed PCA model
with open(os.path.join(SAMS_PATH, "hexamer_pca.pkl"), "rb") as f:
    pca = pickle.load(f)



    
def extract_energy(peptoid_string, std=False, nblocks=5):
    """
    Extracts the probability a peptoid is in the PPII region.

    Args:
        peptoid_string (str): The name of the peptoid (e.g., "PPGPPG").

    Returns:
        float: The extracted probability.
    """
    xtc_path = os.path.join(SAMS_PATH, peptoid_string, "2_sams", "ee_whole.xtc")
    pdb_path = os.path.join(SAMS_PATH, peptoid_string, "2_sams", "ee_whole.pdb")
    state_path = os.path.join(SAMS_PATH, peptoid_string, "2_sams", "alchemical_output_full.txt")
    lam_path = os.path.join(SAMS_PATH, peptoid_string, "2_sams", "mbar_weights_recent.txt")
    states = np.loadtxt(state_path)[:, 2]
    lams = np.loadtxt(lam_path)[:, 1]
    weights = np.exp(-1 * lams[states.astype(np.int32)])
    
    traj = md.load(xtc_path, top=pdb_path)
    weights = weights[:traj.n_frames]

    traj_bkb = traj.atom_slice(traj.top.select("name CA or name CLP or name NL or name OL"))
    dists = compute_all_distances(traj_bkb)

    dist_transform = pca.transform(dists)
    
    probs, xedges, yedges = np.histogram2d(dist_transform[:, 0], dist_transform[:, 1], bins=30, weights=weights)
    probs = probs / np.sum(probs)
    xedges = xedges[1:]
    yedges = yedges[1:]
    stable_region = probs[xedges > 1, :][:, np.abs(yedges) < 1]

    return np.sum(stable_region)

def extract_energy_blockwise(peptoid_string, blocks=5):
    """
    Extracts the energy of a peptoid as in the method above in blocks statistical purposes.

    Args:
        peptoid_string (str): The name of the peptoid (e.g., "PPGPPG").
        blocks (int, optional): The number of blocks to divide the trajectory into. Defaults to 5.

    Returns:
        np.ndarray: An array of energies for each block.
    """
    xtc_path = os.path.join(SAMS_PATH, peptoid_string, "2_sams", "ee_whole.xtc")
    pdb_path = os.path.join(SAMS_PATH, peptoid_string, "2_sams", "ee_whole.pdb")
    state_path = os.path.join(SAMS_PATH, peptoid_string, "2_sams", "alchemical_output_full.txt")
    lam_path = os.path.join(SAMS_PATH, peptoid_string, "2_sams", "mbar_weights_recent.txt")
    states = np.loadtxt(state_path)[:, 2]
    lams = np.loadtxt(lam_path)[:, 1]
    weights = np.exp(-1 * lams[states.astype(np.int32)])
    
    traj = md.load(xtc_path, top=pdb_path)
    weights = weights[:traj.n_frames]

    traj_bkb = traj.atom_slice(traj.top.select("name CA or name CLP or name NL or name OL"))
    dists = compute_all_distances(traj_bkb)

    stabilities = np.zeros(blocks)
    dist_transform = pca.transform(dists)
    n_frames = dist_transform.shape[0]
    for i in range(blocks):
        st = i * n_frames // blocks
        fi = (i + 1) * n_frames // blocks
        probs, xedges, yedges = np.histogram2d(dist_transform[st:fi, 0], dist_transform[st:fi, 1], bins=30, weights=weights[st:fi])
        probs = probs / np.sum(probs)
        xedges = xedges[1:]
        yedges = yedges[1:]

        stable_region = probs[xedges > 1, :][:, np.abs(yedges) < 1]
    
        stabilities[i] = np.sum(stable_region)
    del traj
    del dists
    del dist_transform
    del lams
    del weights
    gc.collect()
    return stabilities

def all_sams_files():
    """
    Finds all valid SAMS files for analysis.

    Returns:
        list: A list of valid SAMS file names.
    """
    files = []
    for file in os.listdir(SAMS_PATH):
        if len(file) == 6 and file[2] == 'G' and file[5] == 'G' and file[0] != 'G':
            if os.path.isfile(os.path.join(SAMS_PATH, file, "2_sams", "ee_whole.pdb")):
                files.append(file)
    return files

def extract_energy_from_all_sams_files(blockwise=False, n_blocks=5):
    """
    Extracts energies for all valid SAMS files.

    Returns:
        tuple: A tuple containing peptoid names and their corresponding energies.
    """
    files = all_sams_files()
    if blockwise:
        energy_f = lambda x: extract_energy_blockwise(x, blocks=n_blocks)
    else:
        energy_f = extract_energy
    energies = list(map(extract_energy, files))
    return files, np.array(energies)


def extract_energy_from_round(i, blockwise=False, n_blocks=5):
    """
    Extracts energies for peptoids in a specific round from a CSV file.

    Args:
        i (int): The round number.

    Returns:
        np.ndarray: An array of energies for the peptoids in the round.
    """
    all_peps = pd.read_csv('bo_peptoids.csv', index_col=0)
    round = all_peps.iloc[i].values.flatten().tolist()
    round = [s for s in round if isinstance(s, str)]
    round = [s for s in round if s[-1] == 'G']
    if blockwise:
        energy_f = lambda x: extract_energy_blockwise(x, blocks=n_blocks)
    else:
        energy_f = extract_energy
    energies = list(map(energy_f, round))
    return np.array(energies)



def save_peptoids(peptoids, i):
    """
    Saves a list of peptoids to a CSV file for a specific round.

    Args:
        peptoids (list): A list of peptoid names
        i (int): The round number.
    """
    print(i)
    all_peps = pd.read_csv("bo_peptoids.csv", index_col=0)
    for n, pep in enumerate(peptoids):
        all_peps.loc[:, "pep" + str(n+1)].iloc[i] = pep

    all_peps.to_csv("bo_peptoids.csv")
    
def save_energies_and_peptoids(peptoids, i):
    """
    Saves both peptoids and their corresponding energies to CSV files for a specific round.

    Args:
        peptoids (list): A list of peptoid names.
        i (int): The round number.
    """

    energies = np.zeros(len(peptoids))
    for n, pep in enumerate(peptoids):
        energies[n] = extract_energy(pep)
    save_peptoids(peptoids, i)
    save_energies(energies, i)

def save_energies(energies, i):
    """
    Saves a list of energies to a CSV file for a specific round.

    Args:
        energies (list): A list of stability energy values.
        i (int): The round number.
    """
    all_energies = pd.read_csv("bo_energies.csv", index_col=0)
    for n, e in enumerate(energies):
        all_energies.loc[:, "e" + str(n+1)].iloc[i] = e
    all_energies.to_csv("bo_energies.csv")

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

def load_model(round, filename='optimizers/optimizer'):
    """
    This function loads a pickled optimization model from a file.

    Args:
        round (int): The round number.
        filename (str, optional): The base filename of the model file. Defaults to 'optimizers/optimizer'.

    Returns:
        object: The loaded optimization model.
    """

    openfile = filename + "_" + str(round) + ".pkl"
    with open(openfile, 'rb') as f:
        return pickle.load(f)


def save_model(optimizer, round, filename='optimizers/optimizer'):
    """
    This function saves a pickled optimization model to a file.

    Args:
        optimizer (object): The optimization model to save.
        round (int): The round number.
        filename (str, optional): The base filename of the model file. Defaults to 'optimizers/optimizer'.
    """
    openfile = filename + "_" + str(round) + ".pkl"
    with open(openfile, 'wb') as f:
        pickle.dump(optimizer, f)
        
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

def get_energies_from_round(i):
    """
    This function reads the "bo_energies.csv" file, extracts the energies for the specified round, and filters out any NaN values.

    Args:
        i (int): The round number.

    Returns:
        list: A list of energies for the specified round.
    """
    all_energies = pd.read_csv('bo_energies.csv', index_col=0)
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






def extract_known_energy(peptoid):
    prev_peps = list(np.concatenate([[p for p in get_peptoids_from_round(r)] for r in range(ROUND)]))
    prev_energies = list(np.concatenate([[p for p in get_energies_from_round(r)] for r in range(ROUND)]))
    return prev_energies[prev_peps.index(peptoid)]

def record_score(round, sc):
    """
    This function loads the scores from a file, updates the score for the current round if specified, and saves the scores back to the file. 

    Args:
        round (int): The current round.
        sc (float): The score for the current round. Defaults to None.
    """
    scores = np.load('scores.npy')
    scores[round] = sc
    np.save('scores.npy', scores)