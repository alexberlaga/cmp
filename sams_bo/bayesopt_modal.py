import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
import itertools
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import pandas as pd
from modAL.models import BayesianOptimizer
from modAL.acquisition import max_EI
import pickle
from scipy import stats
from energy_utils import *
from encoding import CHG, BULKY, ALL_PEPS

MAX_ROUNDS = 40
N_PC = 10
N_QUERY = 7
N_PC = 10
X_ALL = torch.Tensor(np.load('fingerprints_pca.npy')[:, :N_PC])
X_ALL = X_ALL.reshape(1, X_ALL.shape[0], X_ALL.shape[1])





def _query_unique_modal(optimizer, X_all, round, n_instances, tradeoff):
    """
        This function queries for unique data points from a larger dataset using an optimizer.
        
        Args:
        optimizer: An object implementing a query selection strategy.
        X_all: The entire dataset from which to query.
        round: The current round of querying (used for tracking previously queried points).
        n_instances: The desired number of unique data points to query.
        tradeoff: A parameter used by the optimizer to balance different selection criteria.
        
        Returns:
        A list of indices of unique data points queried from X_all.
    """
    # Get indices of previously queried data points from all rounds.
    prev_queried = np.concatenate([[ALL_PEPS.index(p) for p in get_peptoids_from_round(r)] for r in range(round+1)])

    # Initialize the number of queries to make.
    query_num = n_instances
    min_dist = 0
    # Query for the desired number of data points using the optimizer.
    query_idxs = optimizer.query(X_all, n_instances=MAX_QUERY, tradeoff=tradeoff)[0]
    # Loop until we have enough unique queries.
    query_attempt = 0
    true_query = []
    while len(true_query) < n_instances:
        if query_idxs[query_attempt] not in prev_queried:
            true_query.append(query_idxs[query_attempt])
            if len(true_query) >= 4:
                all_combos = itertools.combinations(true_query, 4)
                min_dist = np.min([np.sum(pdist(X_all[combo, :])) for combo in all_combos])
                if min_dist < 25:
                    true_query.pop()
        query_attempt += 1
    print(query_attempt)

            
        
    
    # Return only the indices of unique data points from the final query.
    return true_query

def bayes_round_botorch(peptoids, round=ROUND, n_query=N_QUERY, tradeoff=1, alpha=0.05):
    """
    This function performs a Bayesian optimization round, using BoTorch. It queries the optimizer, suggests new peptoids, evaluates them, updates the optimizer, and saves the results.

    Args:
        petpoids (list): A list of peptoids for the current round.
        round (int, optional): The current round number. Defaults to the global variable ROUND.
        n_query (int, optional): The number of new peptoids to suggest. Defaults to N_QUERY.
        tradeoff (float, optional): The tradeoff between exploration and exploitation in the query strategy. Defaults to 1.

    Returns:
        tuple: A tuple containing the indices of the queried peptoids and the suggested peptoids.
    """
    pass

def bayes_round_modal(peptoids, round=ROUND, n_query=N_QUERY, tradeoff=1, alpha=0.05):
    """
    This function performs a Bayesian optimization round. It queries the optimizer, suggests new peptoids, evaluates them, updates the optimizer, and saves the results.

    Args:
        petpoids (list): A list of peptoids for the current round.
        round (int, optional): The current round number. Defaults to the global variable ROUND.
        n_query (int, optional): The number of new peptoids to suggest. Defaults to N_QUERY.
        tradeoff (float, optional): The tradeoff between exploration and exploitation in the query strategy. Defaults to 1.

    Returns:
        tuple: A tuple containing the indices of the queried peptoids and the suggested peptoids.
    """
    fingerprints_pca = np.load('fingerprints_pca.npy')
    X_all = fingerprints_pca[:, :N_PC]
    save_peptoids(peptoids, round)
    optimizer = load_model(round-1)
    round_idxs = np.zeros(len(peptoids), dtype=np.int32)
    energies = np.zeros(len(peptoids))
    for i in range(len(peptoids)):
        round_idxs[i] = ALL_PEPS.index(peptoids[i])
        energies[i] = extract_energy(peptoids[i])

    
    score(round, energies)    
    optimizer.teach(X_all[round_idxs, :], energies)
    save_energies(energies, round)
    save_model(optimizer, round)
    ROUND = get_current_round()
    round_before_idxs = np.concatenate([[ALL_PEPS.index(p) for p in get_peptoids_from_round(r)] for r in range(round+1)])
    query_idx = _query_unique_modal(optimizer, X_all, round, n_instances=N_QUERY, tradeoff=tradeoff)
    
    save_peptoids([ALL_PEPS[q] for q in query_idx], round + 1)
    return query_idx, [ALL_PEPS[q] for q in query_idx]


def plot_round_modal(round=ROUND, pc=(0, 1), alpha=0.05, n_peptoids=N_QUERY, tradeoff=0):
    """
    This function plots the predicted and true energies of the peptoids for a specified round, as well as comparing them and scoring .

    Args:
        round (int): The round number.
        pc: what PCs to use as axes
    """
    pcx, pcy = pc
    fingerprints_pca = np.load('fingerprints_pca.npy')
    opt = load_model(round, filename="optimizers/optimizer")
    opt2 = load_model(round-1, filename="optimizers/optimizer")

    Xt = opt.estimator.X_train_
    yt = opt.estimator.y_train_
    
    # plt.imshow(opt.estimator.kernel_(opt.estimator.X_train_))
    # plt.show()

    predicted_means, predicted_std = opt.predict(fingerprints_pca[:, :N_PC], return_std=True)
    predicted_means2, predicted_std2 = opt2.predict(fingerprints_pca[:, :N_PC], return_std=True)

    plt.scatter(fingerprints_pca[:, pcx], fingerprints_pca[:, pcy], c=predicted_means, s=10, label='peptoid space')
    plt.colorbar()
    plt.xlabel("PC0", fontsize=20)
    plt.ylabel("PC1", fontsize=20)
    round_idxs = [ALL_PEPS.index(p) for p in get_peptoids_from_round(round)]
    round_before_idxs = np.concatenate([[ALL_PEPS.index(p) for p in get_peptoids_from_round(r)] for r in range(round)])
    plt.title("Predicted probabilities", fontsize=25)
    plt.scatter(fingerprints_pca[round_idxs, pcx], fingerprints_pca[round_idxs, pcy], c="red", label='selected', edgecolors='black', s=20)
    plt.scatter(fingerprints_pca[round_before_idxs, pcx], fingerprints_pca[round_before_idxs, pcy], c=predicted_means[round_before_idxs], label='previous', edgecolors='black', s=20)
    print(np.sort(predicted_means)[-10:])
    plt.tick_params(labelsize=20, size=15)
    plt.show()
    nbins = 25
    mean_probs = np.zeros((nbins, nbins))
    query_idx = _query_unique_modal(opt, fingerprints_pca[:, :N_PC], round, n_instances=n_peptoids, tradeoff=tradeoff)
    _, xedges, yedges = np.histogram2d(fingerprints_pca[:, pcx], fingerprints_pca[:, pcy], bins=nbins)        
    xbins = np.searchsorted(xedges, fingerprints_pca[:, pcx]) - 1
    ybins = np.searchsorted(yedges, fingerprints_pca[:, pcy]) - 1
    for i in range(nbins):
        for j in range(nbins):
            a = np.where(np.logical_and(xbins == i, ybins == j), predicted_means, 0)
            mean_probs[j, i] = np.mean(a[a != 0])
    mean_probs[np.isnan(mean_probs)] = np.min(mean_probs[~np.isnan(mean_probs)])
   
    plt.contourf(mean_probs, levels=30, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar()
    plt.scatter(fingerprints_pca[round_idxs, pcx], fingerprints_pca[round_idxs, pcy], c="red", label='selected', edgecolors='black', s=20)
    plt.scatter(fingerprints_pca[query_idx, pcx], fingerprints_pca[query_idx, pcy], c="pink", label='query', edgecolors='black', s=20)

    plt.scatter(fingerprints_pca[round_before_idxs, pcx], fingerprints_pca[round_before_idxs, pcy], c=predicted_means[round_before_idxs], label='previous', edgecolors='black', s=20)
    plt.xlabel("PC0", fontsize=20)
    plt.ylabel("PC1", fontsize=20)
    plt.title("Bayesian optimization: selecting round " + str(round), fontsize=20)
    plt.tick_params(labelsize=20, size=15)
    plt.legend()
    plt.show()
    
    true_energies = get_energies_from_round(round)
    x = np.arange(len(true_energies))
    width = .4
    plt.bar(x-.2, predicted_means2[round_idxs], width, color='blue', label="Predicted")
    plt.bar(x + .2, true_energies, width, color='red', label="True")
    plt.xticks(x, get_peptoids_from_round(round), rotation=45, ha='right')
    plt.ylabel("PPII Probability", fontsize=20)
    plt.tick_params(labelsize=20, size=15)

    plt.legend()
    plt.show()
    print(predicted_means[query_idx], predicted_std[query_idx], [ALL_PEPS[q] for q in query_idx])
    # score(update=False)

def component_correlation(round=ROUND):
    """
    This function prints the correlations between each PC and helix stability

    Args:
        round (int): The round number.
    """
    fingerprints_pca = np.load('fingerprints_pca.npy')
    correlations = np.zeros(N_PC)
    optimizer = load_model(round-1, filename="optimizers/optimizer")
    predicted_means, predicted_std = optimizer.predict(fingerprints_pca[:, :N_PC], return_std=True)
    all_idxs = np.concatenate([[ALL_PEPS.index(p) for p in get_peptoids_from_round(r)] for r in range(round+1)])
    for i in range(N_PC):
        x = fingerprints_pca[all_idxs, i]
        y = predicted_means[all_idxs]
        coeffs = np.polyfit(x, y, 2)
        # Calculate predicted values
        y_p = np.polyval(coeffs, x)
        
        # Calculate sum of squared residuals (SSres)
        SSres = np.sum((y - y_p)**2)
        
        # Calculate total sum of squares (SStot)
        y_mean = np.mean(y)
        SStot = np.sum((y - y_mean)**2)
        
        # Calculate R^2
        correlations[i] = 1 - SSres / SStot

    return correlations




# def new_bo(round=ROUND):
#     fingerprints_pca = np.load('fingerprints_pca.npy')

#     all_idxs = np.concatenate([[ALL_PEPS.index(p) for p in get_peptoids_from_round(r)] for r in range(round)])
#     Xt = fingerprints_pca[all_idxs, :N_PC]
#     yt = np.concatenate([get_energies_from_round(r) for r in range(round)])
#     print(np.mean(yt))
#     regressor = GaussianProcessRegressor(kernel=kernels.RBF(length_scale=[1]*N_PC, length_scale_bounds=[(.001, 13.9)]*N_PC), n_restarts_optimizer=1000, normalize_y=False, random_state=0, alpha=0.05)
#     optimizer = BayesianOptimizer(estimator=regressor, X_training=Xt, y_training=yt, query_strategy=max_EI)
#     save_model(optimizer, round-1, filename="optimizers/optimizer")

def score(round, true_vals):
    X_all = np.load("fingerprints_pca.npy")[:, :N_PC]
    round_idxs = [ALL_PEPS.index(p) for p in get_peptoids_from_round(round)]
    y_mean = np.mean(np.concatenate([get_energies_from_round(r) for r in range(round)]))
    optimizer = load_model(round - 1)
    y_pred = optimizer.predict(X_all[round_idxs, :])
    # Calculate sum of squared residuals (SSres)
    SSres = np.sum((true_vals - y_pred)**2)
    
    # Calculate total sum of squares (SStot)
    SStot = np.sum((true_vals - y_mean)**2)
    
    # Calculate R^2
    r2 = 1 - SSres / SStot
    record_score(round, r2)
    return r2