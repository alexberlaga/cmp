import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
import itertools
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem.AtomPairs.Pairs import *

from tqdm import tqdm
import itertools
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
# ----------------------------------------------------------------------------------
CHG = 'KXCDE'
TOTAL = 'KXHDECLMFYITVWZ134NQ'
ALL_PEPS = ["".join(r) for r in itertools.product(["".join(s) for s in itertools.product(CHG, repeat=2)], ["".join(s) for s in itertools.product(TOTAL, repeat=2)])]
ALL_PEPS = [p[0] + p[2] + "G" + p[1] + p[3] + "G" for p in ALL_PEPS]





    
def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def seq_encode(peps):
    encoded_list = []
    for pep in peps:
        encode_0 = one_hot_encoding(pep[0], CHG)
        encode_1 = one_hot_encoding(pep[1], TOTAL)
        encode_3 = one_hot_encoding(pep[3], CHG)
        encode_4 = one_hot_encoding(pep[4], TOTAL)
        encoded_list.append(np.concatenate([encode_0, encode_1, encode_3, encode_4]))
    return torch.Tensor(encoded_list)

def get_atom_features(atom, 
                      use_chirality = False, 
                      hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """
    # define list of permitted atoms
    
    permitted_list_of_atoms =  ['C','N','O','S','Br','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-1, 0, 1, "Extreme"])
    
    # hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
        
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + is_in_a_ring_enc + is_aromatic_enc
    # if use_chirality == True:
    #     chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
    #     atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4])
        atom_feature_vector += n_hydrogens_enc
    return np.array(atom_feature_vector)
    
def get_bond_features(bond, 
                      use_stereochemistry = False):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    # if use_stereochemistry == True:
    #     stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
    #     bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)

def make_fingerprints():
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=5)
    fingerprints = np.zeros((len(ALL_PEPS), 2048))
    for i in tqdm(range(len(ALL_PEPS))):
        pep = ALL_PEPS[i]
    
        mol = Chem.MolFromPDBFile("all_pdbs/" + pep + ".pdb")
        fp1 = fpgen.GetFingerprint(mol)
        fingerprints[i] = np.array([b for b in fp1])
        # fingerprints[i, 2048:] = seq_encode([pep])
    
    fingerprints = fingerprints[:, ~np.all(fingerprints == 0, axis=0)]
    if len(np.unique(fingerprints, axis=0)) < len(fingerprints):
        print(len(np.unique(fingerprints, axis=0)))
        print("ERROR")
    np.save("fingerprints.npy", fingerprints)

def make_atompair_fingerprints():
    fingerprints = np.zeros((len(ALL_PEPS), 2048))
    for i in tqdm(range(len(ALL_PEPS))):
        pep = ALL_PEPS[i]
    
        mol = Chem.MolFromPDBFile("all_pdbs/" + pep + ".pdb")
        fp1 = GetHashedAtomPairFingerprint(mol)
        fingerprints[i] = np.array([b for b in fp1])
        # fingerprints[i, 2048:] = seq_encode([pep])
    
    fingerprints = fingerprints[:, ~np.all(fingerprints == 0, axis=0)]
    if len(np.unique(fingerprints, axis=0)) < len(fingerprints):
        print(len(np.unique(fingerprints, axis=0)))
        print("ERROR")
    np.save("ap_fingerprints.npy", fingerprints)
