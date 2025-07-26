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
from tqdm import tqdm


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
    indices = torch.randperm(N, device=device)[:p]
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
        distmin = distance_square[torch.arange(N, device=device), clusters]
        D = distmin.sum().item()

        # Update centers to be the mean of the points in each cluster
        for c in range(p):
            cluster_points = X[clusters == c]
            if len(cluster_points) > 0:
                mu[c] = cluster_points.mean(0)

    # Initialize MPPCA parameters based on k-means results
    # ★★★ 修正点 2: テンソル生成時にdtypeを入力データ(X)に合わせる ★★★
    pi = torch.zeros(p, device=device, dtype=X.dtype)
    W = torch.zeros((p, d, q), device=device, dtype=X.dtype)
    sigma2 = torch.zeros(p, device=device, dtype=X.dtype)

    for c in range(p):
        cluster_mask = (clusters == c)
        num_points = cluster_mask.sum().item()
        
        if num_points == 0:
            # Handle empty clusters if they occur
            pi[c] = 1e-9
            W[c, :, :] = torch.randn(d, q, device=device, dtype=X.dtype) * (variance_level if variance_level is not None else 0.1)
            sigma2[c] = torch.tensor(variance_level if variance_level is not None else 1.0, device=device, dtype=X.dtype)
            continue

        pi[c] = num_points / N

        if variance_level is not None:
            W[c, :, :] = variance_level * torch.randn(d, q, device=device, dtype=X.dtype)
            sigma2[c] = torch.abs((variance_level / 10) * torch.randn(1, device=device, dtype=X.dtype))
        else:
            W[c, :, :] = torch.randn(d, q, device=device, dtype=X.dtype)
            sigma2[c] = (distmin[cluster_mask].mean() / d) + 1e-6

    return pi, mu, W, sigma2, clusters


def mppca_gem_torch(X, pi, mu, W, sigma2, niter, batch_size=1024, device='cpu'):
    """
    Performs the Generalized Expectation-Maximization (GEM) algorithm for MPPCA.
    """
    N, d = X.shape
    p, _, q = W.shape
    
    epsilon = 1e-9

    # Move all parameters to the specified device
    # ★★★ 修正点 3: 渡されたパラメータもdtypeを統一 ★★★
    X = X.to(device, dtype=torch.float64)
    pi = pi.to(device, dtype=torch.float64)
    mu = mu.to(device, dtype=torch.float64)
    W = W.to(device, dtype=torch.float64)
    sigma2 = sigma2.to(device, dtype=torch.float64)

    # ★★★ 修正点 4: 内部で生成するテンソルもdtypeを統一 ★★★
    sigma2hist = torch.zeros((p, niter), device=device, dtype=X.dtype)
    L = torch.zeros(niter, device=device, dtype=X.dtype)
    I_q = torch.eye(q, device=device, dtype=X.dtype)

    pbar = tqdm(range(niter), desc="GEM Algorithm Progress")
    for i in pbar:
        sigma2hist[:, i] = sigma2

        # --- E-step (Expectation) ---
        logR = torch.zeros((N, p), device=device, dtype=X.dtype)
        
        W_T = W.transpose(-2, -1)
        M = sigma2.view(p, 1, 1) * I_q + torch.bmm(W_T, W)
        
        try:
            M_inv = torch.linalg.inv(M)
        except torch.linalg.LinAlgError:
            M_inv = torch.linalg.inv(M + torch.eye(q, device=device, dtype=X.dtype) * epsilon)

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

        # Log-likelihood calculation
        myMax = torch.max(logR, axis=1, keepdim=True).values
        if torch.isinf(myMax).any() or torch.isnan(myMax).any():
             L[i] = torch.tensor(float('nan'), dtype=X.dtype)
        else:
            logR_stable = logR - myMax
            L[i] = (myMax.squeeze() + torch.log(torch.exp(logR_stable).sum(axis=1) + epsilon)).sum()
            L[i] -= N * d * math.log(2 * math.pi) / 2.
        
        pbar.set_postfix(log_likelihood=f"{L[i].item():.4f}")

        # Normalize logR to get R (posterior probabilities)
        logR -= (myMax + torch.log(torch.exp(logR - myMax).sum(axis=1, keepdim=True) + epsilon))
        R = torch.exp(logR)
        if torch.isnan(R).any() or torch.isinf(R).any():
            print("\nWarning: Responsibilities contain nan/inf. Stopping training.")
            L[i:] = float('nan')
            break
            
        # --- M-step (Maximization) ---
        R_sum = R.sum(axis=0)
        R_sum_stable = R_sum + epsilon

        # Update pi
        pi = R_sum / N

        # Update mu
        mu = torch.einsum('np,nd->pd', R, X) / R_sum_stable.unsqueeze(1)

        # Update W and sigma2
        S_W_numerator = torch.zeros((p, d, q), device=device, dtype=X.dtype)
        term2_numerator = torch.zeros(p, device=device, dtype=X.dtype)
        
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            X_batch = X[batch_start:batch_end]
            R_batch = R[batch_start:batch_end]
            
            deviation = X_batch.unsqueeze(0) - mu.unsqueeze(1)
            
            R_dev_T = (R_batch.T.unsqueeze(-1) * deviation).transpose(-2, -1)
            dev_W = torch.bmm(deviation, W)
            S_W_numerator += torch.bmm(R_dev_T, dev_W)
            
            term2_numerator += torch.einsum('np,pnd->p', R_batch, deviation**2)

        S_W = S_W_numerator / R_sum_stable.view(p, 1, 1)
        
        try:
            inner_term = sigma2.view(p, 1, 1) * I_q + torch.bmm(torch.bmm(M_inv, W_T), S_W)
            inner_inv = torch.linalg.inv(inner_term)
        except torch.linalg.LinAlgError:
            inner_term_regularized = inner_term + torch.eye(q, device=device, dtype=X.dtype) * epsilon
            inner_inv = torch.linalg.inv(inner_term_regularized)
            
        W_new = torch.bmm(S_W, inner_inv)

        trace_term = torch.einsum('pdq,pqi->pdi', S_W, M_inv)
        trace_term = torch.einsum('pji,pij->p', W_new.transpose(-2,-1), trace_term)

        sigma2 = (1 / d) * ( (term2_numerator / R_sum_stable) - trace_term )
        sigma2 = torch.clamp(sigma2, min=epsilon)
        
        W = W_new

    pbar.close()
    
    print("\nDone.")
    return pi, mu, W, sigma2, R, L, sigma2hist