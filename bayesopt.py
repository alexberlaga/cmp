import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import pandas as pd
import pickle
from bo_utils import *
from encoding import CHG, BULKY, ALL_PEPS
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP 
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement, qExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from botorch.fit import fit_gpytorch_model
from gpytorch.constraints import Interval
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.penalized import PenalizedAcquisitionFunction, L2Penalty, L1Penalty, GaussianPenalty

MAX_ROUNDS = 40
N_PC = 10
N_QUERY = 8
N_PC = 10
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
    kernel = ScaleKernel(base_kernel=RBFKernel(ard_num_dims=N_PC, lengthscale_constraint=Interval(.001, 13.9), outcome_transform=Standardize(m=1)))
    # Define the model: SingleTaskGP with specified properties

    mean = torch.mean(y_train, dim=1)
    std = torch.std(y_train, dim=1)

    
    y_scaled = (y_train - mean) / std
    yvar_scaled = yvar_train / std
    
    
    
    if yvar_train is None:
        model = SingleTaskGP(train_X=X_train, train_Y=y_train,
                         covar_module=RBFKernel(ard_num_dims=N_PC, lengthscale_constraint=Interval(.001, 13.9), outcome_transform=lambda y: y * std + mean))
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

    fit_gpytorch_model(mll)
    # Train the model for a fixed number of epochs
    # NUM_EPOCHS = 100
    # model.train()
    # for epoch in range(NUM_EPOCHS):
    #     # clear gradients
    #     optimizer.zero_grad()
    #     # forward pass through the model to obtain the output MultivariateNormal
    #     output = model(X_train)
    #     # Compute negative marginal log likelihood
    #     loss = -mll(output, model.train_targets)
    #     # back prop gradients
    #     loss.backward()
    #     # print every 10 iterations
    #     if (epoch + 1) % 10 == 0:
    #         print(
    #             f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} "
    #         )
    #     optimizer.step()
    # Set the model to evaluation mode
    model.eval()  
    # Print the learned lengthscales of the RBF kernel
    print("Fit parameters: ", model.covar_module.lengthscale ) #covar_module_multi.module.base_kernel.lengthscale)
    # Return the trained GPR model
    return model



def base_acquisition(model, y_train, mc=False, tradeoff=-0.3):
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
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]), seed=0)
        MC_EI = qExpectedImprovement(model, best_f=y_train.max() + tradeoff, sampler=sampler)
        torch.manual_seed(seed=0)  # to keep the restart conditions the same
        return MC_EI
    else:
        return ExpectedImprovement(model=model, best_f=y_train.max() + tradeoff)

def sample_and_penalize(acq, X_all, regularization_parameter, sigmoid_param, round, record=False):
    """
    This function samples a point with high acquisition value and penalizes 
    subsequent acquisitions in similar regions.
    
    Args:
      acq (AcquisitionFunction): The acquisition function to be optimized.
      X_all (torch.Tensor): All possible input points (features).
      regularization_parameter (float, optional): Controls the strength of the penalty 
                                                  (defaults to 0.1).
      sigmoid_param (float, optional): Controls the shape of the penalty function 
                                       (defaults to 0.04).
      record: (bool): whether to record the max EI
    
    Returns:
      tuple: A tuple containing three elements:
          - new_idx (int): Index of the selected point in X_all.
          - max_acq (float): Maximum value of the acquisition function.
          - new_acq (AcquisitionFunction): Penalized acquisition function.
    """
    # Get the dimensionality of the data
    d = X_all.shape[-1]
    # Evaluate the acquisition function for all points in X_all
    y_acq = acq(X_all.view(-1, 1, d)).flatten().detach().numpy()
    
    # Get the maximum acquisition value
    max_acq = y_acq.max()
    print(f"Maximum Acquisition Value: {max_acq}")
    
    if record:
        plt.plot(y_acq)
        plt.show()
        print(len(y_acq[y_acq < .05]))
        record_maxEI(max_acq, round)
    
    # Get the index of the point with the highest acquisition value and the point associated with it
    new_idx = np.argmax(y_acq)
    new_point = X_all.view(-1, d)[new_idx]

    # Define a penalty function based on L2 distance and sigmoid function
    penalty = lambda x:  max_acq - max_acq * torch.sigmoid(sigmoid_param * (L2Penalty(init_point=new_point)(x) - 3)) 

    # Create a penalized acquisition function with the defined penalty
    new_acq = PenalizedAcquisitionFunction(raw_acqf=acq, penalty_func=penalty, regularization_parameter=regularization_parameter)

    # Return the index of the selected point, maximum acquisition value, and the penalized acquisition function
    return new_idx, max_acq, new_acq
    


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
        # y1_acq = acq(X_ALL.view(-1, 1, d)).flatten().detach().numpy()
        # y2_acq = new_acq(X_ALL.view(-1, 1, d)).flatten().detach().numpy()
        # plt.scatter(np.linalg.norm(X_all - X_all[new_idx, :], axis=1), y2_acq - y1_acq)
        # plt.show()

        # Update the acquisition function with the penalty based on the selected point
        j += 1
        acq = new_acq

    # Return the array of proposed point indices (casted to int32)
    return propose_idxs.astype(np.int64)
        
def bayes_round(peptoids, round=ROUND, n_query=N_QUERY, tradeoff=-0.3, rp=1, sp=0.2):
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
    save_peptoids(peptoids, round)
    
    # Load the model from the previous round.
    model = load_model(round - 1)
    
    # Get indices of the peptoids in the global ALL_PEPS list.
    round_idxs = [ALL_PEPS.index(pep) for pep in peptoids]
    
    # Extract melting temperatures (Tm) and noise from the current round data.
    tms, noise = extract_tm_from_round(round)

    # Save the extracted melting temperatures and noise data for this round.
    save_tm(tms, round)
    save_noise(noise, round)
    
    # Prepare training data for the new model
    past_idxs = get_idxs_up_to(round)
    new_X_train = X_ALL[past_idxs, :]
    new_y_train = torch.Tensor(get_tm_up_to(round)).reshape(new_X_train.shape[0], 1)    
    new_yvar_train = torch.Tensor(get_noise_up_to(round)).reshape(new_X_train.shape[0], 1)

    # Calculate and log the score of the current model on the new training data.
    score(model, new_X_train, new_y_train, n_peps=len(tms))

    # Create a new Gaussian Process Regression (GPR) model using the prepared training data and save it.
    new_model = init_GPR(new_X_train, new_y_train, new_yvar_train)
    save_model(new_model, round)
    
    # Propose new peptoids for evaluation using the acquisition function with the specified tradeoff, regularization, and shape parameters.
    proposed = propose_points(new_model, new_y_train, n_query, past_idxs, round, tradeoff=tradeoff, rp=rp, sp=sp)
    new_peptoids = [ALL_PEPS[q] for q in proposed]
    save_peptoids(new_peptoids, round + 1)

    # Update the round counter
    update_round()
    
    # Return the indices of the queried peptoids and the list of newly suggested peptoids.
    return proposed, new_peptoids

def bayes_round_nosave(round=ROUND, n_query=N_QUERY, tradeoff=-0.3, rp=1, sp=0.2):
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
    

    peptoids = get_peptoids_from_round(round)
    
    # Load the model from the previous round.
    model = load_model(round - 1)
    
    # Get indices of the peptoids in the global ALL_PEPS list.
    round_idxs = [ALL_PEPS.index(pep) for pep in peptoids]
    

    
    # Prepare training data for the new model
    past_idxs = get_idxs_up_to(round)
    new_X_train = X_ALL[past_idxs, :]
    new_y_train = torch.Tensor(get_tm_up_to(round)).reshape(new_X_train.shape[0], 1)    
    new_yvar_train = torch.Tensor(get_noise_up_to(round)).reshape(new_X_train.shape[0], 1)

   
    # Create a new Gaussian Process Regression (GPR) model using the prepared training data.
    new_model = init_GPR(new_X_train, new_y_train, new_yvar_train)
    
    # Propose new peptoids for evaluation using the acquisition function with the specified tradeoff, regularization, and shape parameters.
    proposed = propose_points(new_model, new_y_train, n_query, past_idxs, round, tradeoff=tradeoff, rp=rp, sp=sp, save=False)
    new_peptoids = [ALL_PEPS[q] for q in proposed]

    
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
    prev_model = load_model(round-1)
    prev_predicted_means = prev_model.posterior(X_ALL).distribution.loc.detach().numpy()

    model = load_model(round)
    # predicted_means = np.diag(model.posterior(X_ALL).distribution.covariance_matrix.detach().numpy())
    predicted_means = model.posterior(X_ALL).distribution.loc.detach().numpy()

    
    pcx, pcy = pc
    ### PLOT 1: PC HEATMAP
    
    nbins = 25
    mean_vals = np.zeros((nbins, nbins))
    _, xedges, yedges = np.histogram2d(X_ALL[:, pcx], X_ALL[:, pcy], bins=nbins)        
    xbins = np.searchsorted(xedges, X_ALL[:, pcx]) - 1
    ybins = np.searchsorted(yedges, X_ALL[:, pcy]) - 1
    for i in range(nbins):
        for j in range(nbins):
            a = np.where(np.logical_and(xbins == i, ybins == j), predicted_means, 0)
            mean_vals[j, i] = np.mean(a[a != 0])
    mean_vals[np.isnan(mean_vals)] = np.min(mean_vals[~np.isnan(mean_vals)])
   
    plt.contourf(mean_vals, levels=30, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar()
    
    plt.scatter(X_ALL[prev_idxs, pcx], X_ALL[prev_idxs, pcy], c=predicted_means[prev_idxs], label='previous', edgecolors='black', s=20)
    plt.scatter(X_ALL[proposed_idxs, pcx], X_ALL[proposed_idxs, pcy], c="red", label='proposed', edgecolors='black', s=20)
    plt.xlabel("PC" + str(pcx), fontsize=20)
    plt.ylabel("PC" + str(pcy), fontsize=20)
    plt.title("Bayesian optimization: selecting round " + str(round+1), fontsize=20)
    plt.tick_params(labelsize=20, size=15)
    plt.legend()
    plt.show()
    
    ### PLOT 2: ACQUISITION FUNCTION HEATMAP
    acq = base_acquisition(model, get_tm_up_to(round), mc=False, tradeoff=0)
    y_acq = acq(X_ALL.view(-1, 1, N_PC)).flatten().detach().numpy()
    mean_vals = np.zeros((nbins, nbins))
    for i in range(nbins):
        for j in range(nbins):
            a = np.where(np.logical_and(xbins == i, ybins == j), y_acq, 0)
            mean_vals[j, i] = np.mean(a[a != 0])
    mean_vals[np.isnan(mean_vals)] = np.min(mean_vals[~np.isnan(mean_vals)])
   
    plt.contourf(mean_vals, levels=30, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar()
    
    plt.scatter(X_ALL[prev_idxs, pcx], X_ALL[prev_idxs, pcy], c="blue", label='previous', edgecolors='black', s=20)
    plt.scatter(X_ALL[proposed_idxs, pcx], X_ALL[proposed_idxs, pcy], c="red", label='proposed', edgecolors='black', s=20)
    plt.xlabel("PC" + str(pcx), fontsize=20)
    plt.ylabel("PC" + str(pcy), fontsize=20)
    plt.title("Bayesian optimization: selecting round " + str(round+1), fontsize=20)
    plt.tick_params(labelsize=20, size=15)
    plt.legend()
    plt.show()
    ### PLOT 3: ACCURACY OF CURRENT ROUND PREDICTIONS

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