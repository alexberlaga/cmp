# Active Learning for the Design of Collagen-Mimetic Peptoids with Thermostable Triple Helical Tertiary Structures   

Code used for the active learning cycle that optimizes the sequence with the most stable possible collagen-like triple helical structure, as measured by the simulated-annealing-calculated melting temperature of the triple helix.   

## REPOSITORY CONTENTS
	
1. bayesopt.py: Code to create a Gaussian Process Regressor, Monte-Carlo Acqusition Function, and propose a certain number of new candidates. Includes code to score the GPR and create analysis plots of the performance of each round of the cycle.   
2. bo_peptoids.csv, bo_tm.csv, and bo_noise.csv: the peptoids selected in each round of the BO cycle, as well as their respective melting temperatures and the standard error of each of the temperatures. 
3. bo_utils.py: Helper functions for Bayesian Optimization and I/O.   
4. check_finished.py: Code to check whether a certain round of simulation is finished.   
5. encoding.py: Code to create atom-pair fingerprints for every peptoid within our design space.   
6. fingerprints.npy and fingerprints_pca.pt: Respectively, a numpy file containing atom-pair fingerprints of each peptoid within our design space, and a PyTorch Tensor with the top-10 principal compoenent embeddings of each of these fingerpints.   
7. generate.py: code to generate PDB files of all the peptoids within our design space. Uses the structure generator I created in the repo https://github.com/UWPRG/mftoid-rev-residues.
8. tm_bo.ipynb: notebook I use to actually run the rounds of Bayesian Optimization.   
