# PyTorch implementation of the algorithm described in
# "Mixtures of Probabilistic Principal Component Analysers",
# Michael E. Tipping and Christopher M. Bishop, Neural Computation 11(2),
# pp 443–482, MIT Press, 1999
#
# This code is a PyTorch translation and enhancement of the original Python/NumPy
# implementation by Mathieu Andreux and Michel Blancard.
# It incorporates batch processing for scalability and vectorization for speed.

import torch
import math

def initialization_kmeans_torch(X, p, q, variance_level=None, device='cpu'):
    """
    Initializes MPPCA parameters using k-means clustering.

    Args:
        X (torch.Tensor): The dataset of shape (N, d).
        p (int): Number of clusters.
        q (int): Dimension of the latent space.
        variance_level (float, optional): A factor for initializing W and sigma2. Defaults to None.
        device (str): The device to run computations on ('cpu' or 'cuda').

    Returns:
        tuple: (pi, mu, W, sigma2, clusters)
            - pi (torch.Tensor): Proportions of clusters, shape (p,).
            - mu (torch.Tensor): Cluster centers, shape (p, d).
            - W (torch.Tensor): Latent to observation matrices, shape (p, d, q).
            - sigma2 (torch.Tensor): Noise variances for each cluster, shape (p,).
            - clusters (torch.Tensor): Cluster assignment for each data point, shape (N,).
    """
    N, d = X.shape
    X = X.to(device)

    # Randomly select initial cluster centers without replacement
    indices = torch.randperm(N)[:p]
    mu = X[indices]

    clusters = torch.zeros(N, dtype=torch.long, device=device)
    D_old = -2.0
    D = -1.0

    # K-means clustering loop
    while D_old != D:
        D_old = D
        # Assign clusters based on the closest center (Euclidean distance)
        # cdist computes pairwise distances between each point in X and each center in mu
        distance_square = torch.cdist(X, mu) ** 2
        clusters = torch.argmin(distance_square, axis=1)

        # Compute distortion (sum of squared distances to the assigned center)
        distmin = distance_square[torch.arange(N), clusters]
        D = distmin.sum().item()

        # Update centers to be the mean of the points in each cluster
        for c in range(p):
            cluster_points = X[clusters == c]
            if len(cluster_points) > 0:
                mu[c] = cluster_points.mean(0)

    # Initialize MPPCA parameters based on k-means results
    pi = torch.zeros(p, device=device)
    W = torch.zeros((p, d, q), device=device)
    sigma2 = torch.zeros(p, device=device)

    for c in range(p):
        cluster_mask = (clusters == c)
        num_points = cluster_mask.sum().item()
        
        if num_points == 0:
            # Handle empty clusters if they occur
            pi[c] = 1e-9 # small probability
            W[c, :, :] = torch.randn(d, q, device=device) * (variance_level if variance_level else 0.1)
            sigma2[c] = torch.tensor(variance_level if variance_level else 1.0, device=device)
            continue

        pi[c] = num_points / N

        if variance_level:
            W[c, :, :] = variance_level * torch.randn(d, q, device=device)
            sigma2[c] = torch.abs((variance_level / 10) * torch.randn(1, device=device))
        else:
            W[c, :, :] = torch.randn(d, q, device=device)
            # Initialize sigma2 as the mean variance of the data in the cluster
            sigma2[c] = (distmin[cluster_mask].mean() / d) + 1e-6 # 微小な値を加えてゼロになるのを防ぐ

    return pi, mu, W, sigma2, clusters


def mppca_gem_torch(X, pi, mu, W, sigma2, niter, batch_size=1024, device='cpu'):
    """
    Performs the Generalized Expectation-Maximization (GEM) algorithm for MPPCA.

    Args:
        X (torch.Tensor): The dataset of shape (N, d).
        pi, mu, W, sigma2: Initial parameters from initialization.
        niter (int): Number of iterations.
        batch_size (int): Size of minibatches for processing large datasets.
        device (str): The device to run computations on ('cpu' or 'cuda').

    Returns:
        tuple: (pi, mu, W, sigma2, R, L, sigma2hist)
            - pi, mu, W, sigma2: The trained model parameters.
            - R (torch.Tensor): Final responsibility matrix, shape (N, p).
            - L (torch.Tensor): Log-likelihood history, shape (niter,).
            - sigma2hist (torch.Tensor): History of sigma2 values, shape (p, niter).
    """
    N, d = X.shape
    p, _, q = W.shape

    # Move all parameters to the specified device
    X = X.to(device)
    pi, mu, W, sigma2 = pi.to(device), mu.to(device), W.to(device), sigma2.to(device)

    sigma2hist = torch.zeros((p, niter), device=device)
    L = torch.zeros(niter, device=device)
    I_d = torch.eye(d, device=device)
    I_q = torch.eye(q, device=device)

    for i in range(niter):
        print('.', end='')
        sigma2hist[:, i] = sigma2

        # --- E-step (Expectation) ---
        # This step is batched to handle large N without memory overflow
        logR = torch.zeros((N, p), device=device)
        
        # Precompute values that are constant for all batches
        W_T = W.transpose(-2, -1)
        M = sigma2.view(p, 1, 1) * I_q + torch.bmm(W_T, W)
        M_inv = torch.linalg.inv(M)
        
        # Use slogdet for numerical stability: log(det(C)) = log(det(sigma2*I_d - W M_inv W_T))
        # This simplifies to: d*log(sigma2) + log(det(M_inv)) + log(det(sigma2*I_q))
        # log |C_c|^{-1/2} = -0.5 * (d * log(sigma2_c) + logdet(M_c) - q*log(sigma2_c))
        # Using C_c^{-1} = (I - W_c M_c^{-1} W_c^T) / sigma2_c
        log_det_C_inv_half = 0.5 * (torch.linalg.slogdet(M_inv).logabsdet - d * torch.log(sigma2) + q * torch.log(sigma2))

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            X_batch = X[batch_start:batch_end]
            
            deviation = X_batch.unsqueeze(0) - mu.unsqueeze(1)  # Shape: (p, batch_N, d)
            
            # Efficiently compute the quadratic form for all p clusters at once
            # (x-mu)^T C_inv (x-mu) using einsum
            # C_inv = (I - W M_inv W_T) / sigma2
            M_inv_W_T = torch.bmm(M_inv, W_T)
            temp = torch.bmm(deviation, W) # Shape: (p, batch_N, q)
            temp = torch.bmm(temp, M_inv_W_T) # Shape: (p, batch_N, d)
            
            # quadratic_form = sum over d of (deviation * (deviation @ C_inv)) / sigma2
            quadratic_form = (torch.sum(deviation**2, dim=-1) - torch.sum(deviation * temp, dim=-1)) / sigma2.view(p, 1)

            # Calculate log-responsibilities for the batch
            logR[batch_start:batch_end, :] = (
                torch.log(pi)
                + log_det_C_inv_half
                - 0.5 * quadratic_form.T
            )

        # Log-likelihood calculation (log-sum-exp trick for numerical stability)
        myMax = torch.max(logR, axis=1, keepdim=True).values
        L[i] = (myMax.squeeze() + torch.log(torch.exp(logR - myMax).sum(axis=1))).sum()
        L[i] -= N * d * math.log(2 * math.pi) / 2.

        # Normalize logR to get R (posterior probabilities)
        logR -= (myMax + torch.log(torch.exp(logR - myMax).sum(axis=1, keepdim=True)))
        R = torch.exp(logR)

        # --- M-step (Maximization) ---
        R_sum = R.sum(axis=0)  # Shape: (p,)

        # Update pi
        pi = R_sum / N

        # Update mu (using einsum for weighted average)
        mu = torch.einsum('np,nd->pd', R, X) / R_sum.unsqueeze(1)

        # Update W and sigma2
        # Accumulate sums over batches to save memory
        S_W_numerator = torch.zeros((p, d, q), device=device)
        term2_numerator = torch.zeros(p, device=device)
        
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            X_batch = X[batch_start:batch_end]
            R_batch = R[batch_start:batch_end]
            
            deviation = X_batch.unsqueeze(0) - mu.unsqueeze(1) # Shape: (p, batch_N, d)
            
            # Accumulate S_W = sum_n(R_nc * (x_n-mu_c)(x_n-mu_c)^T) W_c
            # (R.T * dev).T @ (dev @ W)
            R_dev_T = (R_batch.T.unsqueeze(-1) * deviation).transpose(-2, -1) # Shape: (p, d, batch_N)
            dev_W = torch.bmm(deviation, W) # Shape: (p, batch_N, q)
            S_W_numerator += torch.bmm(R_dev_T, dev_W)
            
            # Accumulate sum_n(R_nc * ||x_n - mu_c||^2)
            term2_numerator += torch.einsum('np,pnd->p', R_batch, deviation**2)

        S_W = S_W_numerator / R_sum.view(p, 1, 1)
        
        # W_new = S_W (sigma2_old * I + M_inv_old W_old^T S_W)^{-1}
        # Vectorized update for W for all p clusters
        inner_inv = torch.linalg.inv(sigma2.view(p, 1, 1) * I_q + torch.bmm(torch.bmm(M_inv, W_T), S_W))
        W_new = torch.bmm(S_W, inner_inv)

        # Update sigma2
        # sigma2_new = (1/d) * [ (1/N_c) * sum(R_nc ||x_n-mu_c||^2) - Tr(W_new^T S_W M_inv) ]
        trace_term = torch.einsum('pdq,pqi->pdi', S_W, M_inv) # S_W @ M_inv
        trace_term = torch.einsum('pji,pij->p', W_new.transpose(-2,-1), trace_term) # trace(W_new.T @ term)

        sigma2 = (1 / d) * ( (term2_numerator / R_sum) - trace_term )
        sigma2 = torch.clamp(sigma2, min=1e-6)
        
        W = W_new

    print("\nDone.")
    return pi, mu, W, sigma2, R, L, sigma2hist


def mppca_predict_torch(X, pi, mu, W, sigma2, batch_size=1024, device='cpu'):
    """
    Predicts cluster responsibilities for new data using a trained MPPCA model.

    Args:
        X (torch.Tensor): The new data of shape (N, d).
        pi, mu, W, sigma2: The trained model parameters.
        batch_size (int): Size of minibatches for processing large datasets.
        device (str): The device to run computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The responsibility matrix R of shape (N, p).
    """
    N, d = X.shape
    p, _, q = W.shape

    # Move all parameters to the specified device
    X = X.to(device)
    pi, mu, W, sigma2 = pi.to(device), mu.to(device), W.to(device), sigma2.to(device)

    # This function is essentially a batched E-step from the training function.
    logR = torch.zeros((N, p), device=device)
    I_d = torch.eye(d, device=device)
    I_q = torch.eye(q, device=device)

    # Precompute values that are constant for all batches
    W_T = W.transpose(-2, -1)
    M = sigma2.view(p, 1, 1) * I_q + torch.bmm(W_T, W)
    M_inv = torch.linalg.inv(M)
    log_det_C_inv_half = 0.5 * (torch.linalg.slogdet(M_inv).logabsdet - d * torch.log(sigma2) + q * torch.log(sigma2))
    
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        X_batch = X[batch_start:batch_end]
        
        deviation = X_batch.unsqueeze(0) - mu.unsqueeze(1)
        
        M_inv_W_T = torch.bmm(M_inv, W_T)
        temp = torch.bmm(deviation, W)
        temp = torch.bmm(temp, M_inv_W_T)
        quadratic_form = (torch.sum(deviation**2, dim=-1) - torch.sum(deviation * temp, dim=-1)) / sigma2.view(p, 1)

        logR[batch_start:batch_end, :] = (
            torch.log(pi)
            + log_det_C_inv_half
            - 0.5 * quadratic_form.T
        )

    # Normalize logR to get R (posterior probabilities)
    myMax = torch.max(logR, axis=1, keepdim=True).values
    logR -= (myMax + torch.log(torch.exp(logR - myMax).sum(axis=1, keepdim=True)))
    R = torch.exp(logR)

    return R