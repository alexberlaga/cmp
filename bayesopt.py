import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import pandas as pd
import pickle
from bo_utils import *
from encoding import ALL_PEPS
from botorch.optim.optimize import optimize_acqf_discrete
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP 
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement, qExpectedImprovement
from botorch.sampling import IIDNormalSampler
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from botorch.fit import fit_gpytorch_model
from gpytorch.constraints import Interval
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.penalized import PenalizedAcquisitionFunction, L2Penalty, L1Penalty, GaussianPenalty

MAX_ROUNDS = 40
N_PC = 10
N_QUERY = 8
N_PC = 10
SEED = 613
X_ALL = torch.Tensor(np.load('fingerprints_pca.npy')[:, :N_PC])




def init_GPR(X_train, y_train, yvar_train=None):
    """
    This function initializes a Gaussian Process Regression (GPR) model using GPyTorch.
    
    Args:
      X_train (torch.Tensor): Training features.
      y_train (torch.Tensor): Training target values (melting temperatures in this case).
      yvar_train (torch.Tensor, optional): Training noise levels (defaults to None).
    
    Returns:
      SingleTaskGP: A trained SingleTaskGP model from GPyTorch.
    """
    # Define the kernel function: Radial Basis Function (RBF) with scaling
    kernel = ScaleKernel(base_kernel=RBFKernel(ard_num_dims=N_PC, lengthscale_constraint=Interval(.9, 114.5), outcome_transform=Standardize(m=1)))
    # Define the model: SingleTaskGP with specified properties

    mean = torch.mean(y_train, dim=1)
    std = torch.std(y_train, dim=1)

    
    y_scaled = (y_train - mean) / std
    yvar_scaled = yvar_train / std
    
    
    
    if yvar_train is None:
        model = SingleTaskGP(train_X=X_train, train_Y=y_train,
                         covar_module=RBFKernel(ard_num_dims=N_PC, lengthscale_constraint=Interval(.9, 114.5), outcome_transform=lambda y: y * std + mean))
    else:
        model = SingleTaskGP(train_X=X_train, train_Y=y_train, train_Yvar=yvar_train,
                         covar_module=kernel, outcome_transform=Standardize(m=1))
    
    
    # Define the marginal log likelihood for training
    mll = ExactMarginalLogLikelihood(model.likelihood, model) 
    
    # Define the optimizer (Adam)
    optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': model.likelihood.parameters()},], lr=0.05)
    
    mll = mll.to(X_train)

    fit_gpytorch_model(mll, max_retries=1)
    
    # Set the model to evaluation mode
    model.eval()  
    # Print the learned lengthscales of the RBF kernel
    print("Fit parameters: ", model.covar_module.lengthscale ) #covar_module_multi.module.base_kernel.lengthscale)
    # Return the trained GPR model
    return model



def base_acquisition(model, y_train, mc=False, tradeoff=0):
    """
    This function calculates the Expected Improvement (EI) acquisition function for the given model.
    
    Args:
      model (SingleTaskGP): The trained GPR model.
      y_train (torch.Tensor): Training target values (melting temperatures).
      mc (bool, optional): Flag for using Monte Carlo sampling (defaults to False).
      tradeoff (float, optional): Exploration-exploitation tradeoff parameter (defaults to -0.3).
    
    Returns:
      AcquisitionFunction: The calculated EI acquisition function.
    """
    if mc:
        sampler = IIDNormalSampler(sample_shape=torch.Size([2048]), seed=SEED)
        MC_EI = qExpectedImprovement(model, best_f=y_train.max() + tradeoff, sampler=sampler)
        torch.manual_seed(seed=SEED)  # to keep the restart conditions the same
        return MC_EI
    else:
        return ExpectedImprovement(model=model, best_f=y_train.max() + tradeoff)




    

def propose_q_points(model, y_train, n_points, prev_idxs, tradeoff=0, X_all=X_ALL):
    # Initialize an empty array to store proposed point indices
    propose_idxs = np.zeros(n_points)

    # Create the base acquisition function (Expected Improvement)
    acq = base_acquisition(model, y_train, mc=True, tradeoff=tradeoff)
    X_unique = np.delete(X_all, prev_idxs, axis=0)
    mins = torch.min(X_unique, dim=0)
    maxs = torch.max(X_unique, dim=0)
    candidates, acq_vals = optimize_acqf_discrete(acq, q=n_points, choices=X_unique, unique=True)
    print(acq_vals)
    cands = candidates.detach().numpy()
    X_np = X_all.detach().numpy()
    idx_vals = np.zeros(X_np.shape[0])
    for j in range(cands.shape[0]):
        idx_vals += [np.allclose(X_np[i], cands[j]) for i in range(X_np.shape[0])]
    return np.array(np.nonzero(idx_vals)[0]).astype(np.int64)

def propose_points(model, y_train, n_points, prev_idxs, round=ROUND, tradeoff=-0.3, rp=1, sp=0.2, X_all=X_ALL, save=True):
    """
    This function proposes new points for evaluation using an iterative acquisition function optimization.
    
    Args:
      model (SingleTaskGP): The trained GPR model.
      y_train (torch.Tensor): Training target values (melting temperatures).
      n_points (int): Number of points to propose.
      tradeoff (float, optional): Exploration-exploitation tradeoff parameter in the acquisition function (defaults to -0.3).
      rp (float, optional): Regularization parameter for the penalty function (defaults to -0.1).
      sp (float, optional): Shape parameter for the penalty function (defaults to 0.03).
      X_all (torch.Tensor, optional): All possible input points (features) (defaults to X_ALL).
    
    Returns:
      numpy.ndarray: Array containing indices of the proposed points in X_all.
    """
    # Initialize an empty array to store proposed point indices
    propose_idxs = np.zeros(n_points)

    # Create the base acquisition function (Expected Improvement)
    acq = base_acquisition(model, y_train, mc=False, tradeoff=tradeoff)
    d = X_all.shape[-1]
    i = 0
    j = 0
    all_idxs = prev_idxs
    # Iteratively propose points using acquisition function optimization with penalty
    while i < n_points:
        # Sample a point with high acquisition value and penalize similar regions
        if save:
            record = j == 0
        else:
            record = False
        new_idx, acq_val, new_acq = sample_and_penalize(acq, X_all, rp, sp, round, record)
        if new_idx not in all_idxs:
            all_idxs = np.append(all_idxs, new_idx)
            propose_idxs[i] = new_idx
            i += 1
        
        # Update the acquisition function with the penalty based on the selected point
        j += 1
        acq = new_acq

    # Return the array of proposed point indices (casted to int32)
    return propose_idxs.astype(np.int64)
        
def bayes_round(peptoids, round=ROUND, n_query=N_QUERY, q=False, tradeoff=-0.3, rp=1, sp=0.2, save=True):
    """
    This function performs a Bayesian optimization round, using BoTorch. It queries the optimizer, suggests new peptoids, evaluates them, updates the optimizer, and saves the results.

    Args:
        petpoids (list): A list of peptoids for the current round.
        round (int, optional): The current round number. Defaults to the global variable ROUND.
        n_query (int, optional): The number of new peptoids to suggest. Defaults to N_QUERY.
        kwargs:
            tradeoff (float, optional): addition to f_max in EI acquisition function. Positive: more exploration, negative: more exploitation.
            rp (float, optional): The regularization parameter: how much penalty contributes to acquisition function. 
            sp (float, optional): Controls the shape of the sigmoid in the penalty. The higher, the sharper the penalty function.
    Returns:
        tuple: A tuple containing the indices of the queried peptoids and the suggested peptoids.
    """
    
    # Save the peptoids evaluated in this round.
    if save:
        save_peptoids(peptoids, round)
    
    # Load the model from the previous round.
    if round > 0:
        model = load_model(round - 1)
    
    # Get indices of the peptoids in the global ALL_PEPS list.
    round_idxs = [ALL_PEPS.index(pep) for pep in peptoids]
    
    # Extract melting temperatures (Tm) and noise from the current round data.
    tms, noise = extract_tm_from_round(round)

    # Save the extracted melting temperatures and noise data for this round.
    if save:
        save_tm(tms, round)
        save_noise(noise, round)
    
    # Prepare training data for the new model
    past_idxs = get_idxs_up_to(round)
    new_X_train = X_ALL[past_idxs, :]
    new_y_train = torch.Tensor(get_tm_up_to(round)).reshape(new_X_train.shape[0], 1)

    new_yvar_train = torch.Tensor(get_noise_up_to(round)).reshape(new_X_train.shape[0], 1)

    # Calculate and log the score of the current model on the new training data.
    if round > 0:
        score(model, new_X_train, new_y_train, n_peps=len(tms))

    # Create a new Gaussian Process Regression (GPR) model using the prepared training data and save it.
    new_model = init_GPR(new_X_train, new_y_train, new_yvar_train)
    if save:
        save_model(new_model, round)
    
    # Propose new peptoids for evaluation using the acquisition function with the specified tradeoff, regularization, and shape parameters.
    if q:
        proposed = propose_q_points(new_model, new_y_train, n_query, past_idxs, tradeoff=tradeoff, X_all=X_ALL)
    else:
        proposed = propose_points(new_model, new_y_train, n_query, past_idxs, round, tradeoff=tradeoff, rp=rp, sp=sp)
    new_peptoids = [ALL_PEPS[q] for q in proposed]
    if save:
        save_peptoids(new_peptoids, round + 1)

    # Update the round counter
    update_round()
    
    # Return the indices of the queried peptoids and the list of newly suggested peptoids.
    return proposed, new_peptoids



def plot_round(round=ROUND, pc=(0, 1)):
    """
    This function plots the predicted and true energies of the peptoids for a specified round, as well as comparing them and scoring .

    Args:
        round (int): The round number.
        pc: what PCs to use as axes
    """
    proposed_idxs = [ALL_PEPS.index(p) for p in get_peptoids_from_round(round+1)]
    prev_idxs = get_idxs_up_to(round)
    cur_idxs = [ALL_PEPS.index(p) for p in get_peptoids_from_round(round)]
    if round > 0:
        prev_model = load_model(round-1)
        prev_predicted_means = prev_model.posterior(X_ALL).distribution.loc.detach().numpy()

    model = load_model(round)
    predicted_means = model.posterior(X_ALL).distribution.loc.detach().numpy()

    
    pcx, pcy = pc
    ### PLOT 1: PC HEATMAP
    
    nbins = 25
    mean_vals = np.zeros((nbins, nbins))
    xa = X_ALL.detach().numpy()
    _, xedges, yedges = np.histogram2d(xa[:, pcx], xa[:, pcy], bins=nbins)
    xbins = np.searchsorted(xedges, xa[:, pcx]) - 1
    ybins = np.searchsorted(yedges, xa[:, pcy]) - 1
    for i in range(nbins):
        for j in range(nbins):
            a = np.where(np.logical_and(xbins == i, ybins == j), predicted_means, 0)
            if len(a[a != 0]) == 0:
                mean_vals[j, i] = np.nan
            else:
                mean_vals[j, i] = a[a != 0].max()
    if len(np.isnan(mean_vals)) > 0:
        mean_vals[np.isnan(mean_vals)] = np.min(mean_vals[~np.isnan(mean_vals)])
   
    c = plt.contourf(mean_vals, levels=np.linspace(330, 420, 91), extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],  vmin=330, vmax=420, cmap='jet')
    cbar = plt.colorbar(c, ax=plt.gca(), extend='both')
    cbar.ax.set_ylabel('Maximum Expected \n Temperature in Region', fontsize=20)
    
    avail_points, _, _ = np.histogram2d(xa[:, pcx], xa[:, pcy], bins=nbins)
    plt.contour(avail_points.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], levels=[1], alpha=1, linewidths=[2], label='available candidates')


    plt.scatter(xa[prev_idxs, pcx], xa[prev_idxs, pcy], c=predicted_means[prev_idxs], label='previous', edgecolors='black', s=10, cmap='jet')
    plt.scatter(xa[proposed_idxs, pcx], xa[proposed_idxs, pcy], c="red", label='proposed', edgecolors='black', s=20)
    plt.xlabel("PC" + str(pcx), fontsize=20)
    plt.ylabel("PC" + str(pcy), fontsize=20)
    plt.title("Bayesian optimization: selecting round " + str(round+1), fontsize=20)
    plt.tick_params(labelsize=20, size=15)
    plt.legend()
    plt.show()
    
    ### PLOT 2: ACQUISITION FUNCTION HEATMAP
    acq = base_acquisition(model, get_tm_up_to(round), mc=True, tradeoff=0)
    y_acq = np.log(acq(X_ALL.view(-1, 1, N_PC)).flatten().detach().numpy() + np.exp(-5.99))
    print(np.partition(y_acq, -10)[-10:])
    mean_vals = np.zeros((nbins, nbins))
    for i in range(nbins):
        for j in range(nbins):
            a = np.where(np.logical_and(xbins == i, ybins == j), y_acq, 0)
            if len(a[a != 0]) == 0:
                mean_vals[j, i] = np.nan
            else:
                mean_vals[j, i] = a[a != 0].max()
    if len(np.isnan(mean_vals)) > 0:
        mean_vals[np.isnan(mean_vals)] = np.min(mean_vals[~np.isnan(mean_vals)])
   
    c = plt.contourf(mean_vals,  levels=np.linspace(-6, 6, 121), extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='jet', vmin=-5, vmax=5,)
    cbar = plt.colorbar(c, ax=plt.gca(), extend='both')
    
    plt.contour(avail_points.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], levels=[1], alpha=1, colors=['pink'], linewidths=[2],  label='available candidates')

    
    cbar.ax.set_ylabel('Acquisition Function Value', fontsize=20)

    plt.scatter(X_ALL[prev_idxs, pcx], X_ALL[prev_idxs, pcy], c="yellow", label='previous', edgecolors='black',  s=10)
    plt.scatter(X_ALL[proposed_idxs, pcx], X_ALL[proposed_idxs, pcy], c="red", label='proposed', edgecolors='black', s=20)
    plt.xlabel("PC" + str(pcx), fontsize=20)
    plt.ylabel("PC" + str(pcy), fontsize=20)
    plt.title("Bayesian optimization: selecting round " + str(round+1), fontsize=20)
    plt.tick_params(labelsize=20, size=15)
    plt.legend()
    plt.show()
    ### PLOT 3: ACCURACY OF CURRENT ROUND PREDICTIONS
    if round > 0:
        true_tms = get_tm_from_round(round)
        x = np.arange(len(true_tms))
        width = .4
        plt.bar(x-.2, prev_predicted_means[cur_idxs], width, color='blue', label="Predicted")
        plt.bar(x + .2, true_tms, width, color='red', label="True")
        plt.xticks(x, get_peptoids_from_round(round), rotation=45, ha='right')
        plt.ylabel("Melting Temperature (K)", fontsize=20)
        plt.tick_params(labelsize=20, size=15)
        plt.gca().set_ylim(bottom=310)
        plt.legend()
        plt.show()

    ###PLOT 4: R^2 OF THE MODEL FOR EVERY ROUND 
    
    scores = np.load('scores.npy')
       
    plt.plot(scores[:round+1])
    plt.xlabel("Round", fontsize=20)
    plt.ylabel("$R^2$", fontsize=20)
    plt.title("Scores", fontsize=25)
    plt.tick_params(labelsize=20, size=15)
    plt.show()

    ###PLOT 5: ACQUISITION FUNCTION FOR EVERY ROUND 

    eis = np.load('max_eis.npy')
       
    plt.plot(eis[:round+1])
    plt.xlabel("Round", fontsize=20)
    plt.ylabel("Expected Improvement", fontsize=20)
    plt.title("Expected Improvement Each Round", fontsize=25)
    plt.tick_params(labelsize=20, size=15)
    plt.show()

def plot_all_temps(plotsize=[6, 4], pc=[0, 1]):
    px, py = plotsize[0], plotsize[1]
    fig, ax = plt.subplots(px, py, figsize=[3 * px, 6 * py])
    pcx, pcy = pc
    for rnd in range(24):
        ax_ = ax[rnd // py, rnd % py]
        proposed_idxs = [ALL_PEPS.index(p) for p in get_peptoids_from_round(rnd+1)]
        prev_idxs = get_idxs_up_to(rnd)
        cur_idxs = [ALL_PEPS.index(p) for p in get_peptoids_from_round(rnd)]
        if rnd > 0:
            prev_model = load_model(rnd-1)
            prev_predicted_means = prev_model.posterior(X_ALL).distribution.loc.detach().numpy()
    
        model = load_model(rnd)
        predicted_means = model.posterior(X_ALL).distribution.loc.detach().numpy()
    
        
        
        ### PLOT 1: PC HEATMAP
        
        nbins = 25
        mean_vals = np.zeros((nbins, nbins))
        xa = X_ALL.detach().numpy()
        _, xedges, yedges = np.histogram2d(xa[:, pcx], xa[:, pcy], bins=nbins)
        xbins = np.searchsorted(xedges, xa[:, pcx]) - 1
        ybins = np.searchsorted(yedges, xa[:, pcy]) - 1
        for i in range(nbins):
            for j in range(nbins):
                a = np.where(np.logical_and(xbins == i, ybins == j), predicted_means, 0)
                if len(a[a != 0]) == 0:
                    mean_vals[j, i] = np.nan
                else:
                    mean_vals[j, i] = a[a != 0].max()
        if len(np.isnan(mean_vals)) > 0:
            mean_vals[np.isnan(mean_vals)] = np.min(mean_vals[~np.isnan(mean_vals)])
       
        c = ax_.contourf(mean_vals, levels=np.linspace(330, 420, 91), extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],  vmin=330, vmax=420, cmap='jet')
        
        
        avail_points, _, _ = np.histogram2d(xa[:, pcx], xa[:, pcy], bins=nbins)
        ax_.contour(avail_points.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], levels=[1], alpha=1, linewidths=[2], label='available candidates')
    
    
        ax_.scatter(xa[prev_idxs, pcx], xa[prev_idxs, pcy], c=predicted_means[prev_idxs], label='previous', edgecolors='black', s=10, cmap='jet')
        ax_.scatter(xa[proposed_idxs, pcx], xa[proposed_idxs, pcy], c="red", label='proposed', edgecolors='black', s=20)
        ax_.set_xlabel("PC" + str(pcx), fontsize=20)
        ax_.set_ylabel("PC" + str(pcy), fontsize=20)
        ax_.set_title("Selecting round " + str(rnd+1), fontsize=20)
        ax_.tick_params(labelsize=20, size=15)
        ax_.legend()

    plt.tight_layout(pad=2)
    cbar = fig.colorbar(c, ax=ax, fraction=0.07, aspect=40)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(24)
    cbar.ax.set_ylabel('Maximum Expected Temperature in Region', fontsize=35)
    plt.show()

def plot_all_acq(plotsize=[6, 4], pc=[0, 1]):
    px, py = plotsize[0], plotsize[1]
    fig, ax = plt.subplots(px, py, figsize=[3 * px, 6 * py])
    pcx, pcy = pc
    xa = X_ALL.detach().numpy()
    avail_points, _, _ = np.histogram2d(xa[:, pcx], xa[:, pcy], bins=25)
    for rnd in range(24):
        ax_ = ax[rnd // py, rnd % py]
        nbins = 25
        mean_vals = np.zeros((nbins, nbins))
        model = load_model(rnd)
        proposed_idxs = [ALL_PEPS.index(p) for p in get_peptoids_from_round(rnd+1)]
        prev_idxs = get_idxs_up_to(rnd)
        cur_idxs = [ALL_PEPS.index(p) for p in get_peptoids_from_round(rnd)]
        _, xedges, yedges = np.histogram2d(xa[:, pcx], xa[:, pcy], bins=nbins)
        xbins = np.searchsorted(xedges, xa[:, pcx]) - 1
        ybins = np.searchsorted(yedges, xa[:, pcy]) - 1
        acq = base_acquisition(model, get_tm_up_to(rnd), mc=True, tradeoff=0)
        y_acq = np.log(acq(X_ALL.view(-1, 1, N_PC)).flatten().detach().numpy() + np.exp(-5.99))
        mean_vals = np.zeros((nbins, nbins))
        for i in range(nbins):
            for j in range(nbins):
                a = np.where(np.logical_and(xbins == i, ybins == j), y_acq, 0)
                if len(a[a != 0]) == 0:
                    mean_vals[j, i] = np.nan
                else:
                    mean_vals[j, i] = a[a != 0].max()
        if len(np.isnan(mean_vals)) > 0:
            mean_vals[np.isnan(mean_vals)] = np.min(mean_vals[~np.isnan(mean_vals)])
       
        c = ax_.contourf(mean_vals,  levels=np.linspace(-6, 6, 121), extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='jet', vmin=-5, vmax=5,)
        
        ax_.contour(avail_points.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], levels=[1], alpha=1, colors=['pink'], linewidths=[2],  label='available candidates')
    
        
        
        ax_.scatter(xa[prev_idxs, pcx], xa[prev_idxs, pcy], c='yellow', label='previous', edgecolors='black', s=10, cmap='jet')
        ax_.scatter(xa[proposed_idxs, pcx], xa[proposed_idxs, pcy], c="red", label='proposed', edgecolors='black', s=20)
        ax_.set_xlabel("PC" + str(pcx), fontsize=20)
        ax_.set_ylabel("PC" + str(pcy), fontsize=20)
        ax_.set_title("Selecting round " + str(rnd+1), fontsize=20)
        ax_.tick_params(labelsize=20, size=15)
        ax_.legend()
    plt.tight_layout(pad=2)
    cbar = fig.colorbar(c, ax=ax, fraction=0.07, aspect=40)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(24)
    cbar.ax.set_ylabel('Acquisition Function Value', fontsize=35)
    plt.show()

def component_correlation(round=ROUND):
    correlations = np.zeros(N_PC)
    gpr = load_model(round)
    predicted_means = gpr.posterior(X_ALL).distribution.loc.detach().numpy()
    all_idxs = get_idxs_up_to(round)
    for i in range(N_PC):
        x = X_ALL[all_idxs, i]
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

def score(model, X, y_tensor, n_peps=8, round=ROUND):
    y = y_tensor.detach().numpy().flatten()
    y_short = y[-n_peps:]
    y_mean = np.mean(y)
    y_pred = model.posterior(X).distribution.loc.detach().numpy().flatten()[-n_peps:]
    # Calculate sum of squared residuals (SSres)
    SSres = np.sum((y_short - y_pred)**2)
    
    # Calculate total sum of squares (SStot)
    SStot = np.sum((y_short - y_mean)**2)
    
    # Calculate R^2
    r2 = 1 - SSres / SStot
    record_score(round, r2)
    
    return r2

def command_script(peptoids):
    return "bash round_small_melt.sh " + " ".join(peptoids)