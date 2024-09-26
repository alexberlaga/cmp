import itertools
import sys, os
import encoding
from encoding import ALL_PEPS, CHG, TOTAL
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import json
import torch
from torch_geometric.data import Data
import numpy as np
structure_maker_path = "/project2/andrewferguson/berlaga/peptoids/structure_maker"
sys.path.append(structure_maker_path)
import create_peptoid

cur_path = os.getcwd()


pdb_path = os.path.join(cur_path, "all_pdbs")
unrelated_smiles = "O=O"
unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
n_node_features = len(encoding.get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
n_edge_features = len(encoding.get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

def run():
    os.chdir(structure_maker_path)
    data_list = []
    count = 0
    c2 = 1
    for pep in ALL_PEPS:
        fname = pep + ".pdb"
        create_peptoid.create_peptoid(pep, filename=fname)
        mol = Chem.MolFromPDBFile(fname)
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = encoding.get_atom_features(atom)
        X = torch.tensor(X, dtype = torch.float)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        EF = np.zeros((n_edges, n_edge_features))
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            EF[k] = encoding.get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        EF = torch.tensor(EF, dtype = torch.float)
        graph_data = Data(x = X, edge_index = E, edge_attr = EF)
        data_list.append(graph_data)
        if c2 % 100 == 0:
            torch.save(data_list, os.path.join(cur_path, "graphs_{}.pt".format(count)))
            count += 1
            data_list = []
        c2 += 1
        os.remove(pep + ".pdb")
    torch.save(data_list, os.path.join(cur_path, "graphs_{}.pt".format(count)))
    os.chdir(cur_path)

def create_all_pdbs():
    os.chdir(structure_maker_path)
    data_list = []
    count = 0
    c2 = 1
    for pep in ALL_PEPS:
        fname = os.path.join(pdb_path, pep + ".pdb")
        if not os.path.isfile(fname):
            create_peptoid.create_peptoid(pep, filename=fname)
    os.chdir(cur_path)

create_all_pdbs()
