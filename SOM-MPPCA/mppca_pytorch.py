import torch
import math
from tqdm import tqdm


def initialization_kmeans_torch(X, p, q, variance_level=None, device='cpu'):
    """
    Initializes MPPCA parameters using k-means clustering.
    Args:
        X (torch.Tensor): The dataset of shape (N, d). Expected to be on the target device.
        p (int): Number of clusters.
        q (int): Dimension of the latent space.
        variance_level (float, optional): A factor for initializing W and sigma2. Defaults to None.
        device (str): The device to run computations on ('cpu' or 'cuda').

    Returns:
        tuple: (pi, mu, W, sigma2, clusters)
    """
    N, d = X.shape
    # X is assumed to be on the correct device already.

    # Randomly select initial cluster centers without replacement
    indices = torch.randperm(N, device=device)[:p]
    mu = X[indices]

    clusters = torch.zeros(N, dtype=torch.long, device=device)
    D_old = -2.0
    D = -1.0

    # K-means clustering loop
    # This loop is kept simple for initialization. More robust k-means could be used.
    for _ in range(100): # Use a fixed number of iterations to prevent infinite loops
        # Assign clusters based on the closest center (Euclidean distance)
        distance_square = torch.cdist(X, mu) ** 2
        clusters = torch.argmin(distance_square, axis=1)

        # Compute distortion (sum of squared distances to the assigned center)
        distmin = distance_square[torch.arange(N, device=device), clusters]
        current_D = distmin.sum().item()
        
        # Check for convergence
        if D_old == current_D:
            break
        D_old = current_D

        # Update centers to be the mean of the points in each cluster
        for c in range(p):
            cluster_points = X[clusters == c]
            if len(cluster_points) > 0:
                mu[c] = cluster_points.mean(0)
    
    D = distmin.sum().item()

    # Initialize MPPCA parameters based on k-means results
    pi = torch.zeros(p, device=device, dtype=X.dtype)
    W = torch.zeros((p, d, q), device=device, dtype=X.dtype)
    sigma2 = torch.zeros(p, device=device, dtype=X.dtype)

    for c in range(p):
        cluster_mask = (clusters == c)
        num_points = cluster_mask.sum().item()
        
        if num_points == 0:
            # Handle empty clusters if they occur
            pi[c] = 1e-9 # Assign a very small probability
            # Re-initialize mu for the empty cluster to a random point
            mu[c] = X[torch.randint(0, N, (1,)).item()]
            W[c, :, :] = torch.randn(d, q, device=device, dtype=X.dtype) * 0.1
            sigma2[c] = torch.var(X) / d # Use global variance
            continue

        pi[c] = num_points / N

        if variance_level is not None:
            W[c, :, :] = variance_level * torch.randn(d, q, device=device, dtype=X.dtype)
            sigma2[c] = torch.abs((variance_level / 10) * torch.randn(1, device=device, dtype=X.dtype))
        else:
            # Initialize W with small random values
            W[c, :, :] = torch.randn(d, q, device=device, dtype=X.dtype) * 0.1
            # Initialize sigma2 based on the in-cluster variance
            sigma2[c] = (distmin[cluster_mask].mean() / d) + 1e-6

    return pi, mu, W, sigma2, clusters


def mppca_gem_torch(X, pi, mu, W, sigma2, niter, batch_size=1024, device='cpu'):
    """
    Performs the Generalized Expectation-Maximization (GEM) algorithm for MPPCA.
    
    Args:
        X, pi, mu, W, sigma2: Tensors for data and parameters.
        These are expected to be on the target device and of dtype float64 for stability.
    """
    N, d = X.shape
    p, _, q = W.shape
    
    # Epsilon: A small constant to prevent division by zero or log(0)
    epsilon = torch.finfo(X.dtype).eps

    # Tensors for tracking history, created with the same dtype as X
    sigma2hist = torch.zeros((p, niter), device=device, dtype=X.dtype)
    L = torch.zeros(niter, device=device, dtype=X.dtype)
    I_q = torch.eye(q, device=device, dtype=X.dtype)

    pbar = tqdm(range(niter), desc="GEM Algorithm Progress")
    for i in pbar:
        sigma2hist[:, i] = sigma2

        # --- E-step (Expectation) ---
        logR = torch.zeros((N, p), device=device, dtype=X.dtype)
        
        W_T = W.transpose(-2, -1)
        # Add a small value to sigma2 on the diagonal for robustness
        M = (sigma2.view(p, 1, 1) + epsilon) * I_q + torch.bmm(W_T, W)
        
        try:
            M_inv = torch.linalg.inv(M)
            log_det_M = torch.linalg.slogdet(M).logabsdet
        except torch.linalg.LinAlgError:
            # If M is singular, add a small identity matrix for regularization
            M_inv = torch.linalg.inv(M + torch.eye(q, device=device, dtype=X.dtype) * 1e-6)
            log_det_M = torch.linalg.slogdet(M + torch.eye(q, device=device, dtype=X.dtype) * 1e-6).logabsdet

        # Pre-calculate the log determinant term of the inverse covariance
        log_det_C_inv_half = -0.5 * (log_det_M + (d - q) * torch.log(sigma2 + epsilon))

        # Process data in batches to manage memory usage
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            X_batch = X[batch_start:batch_end]
            
            deviation = X_batch.unsqueeze(0) - mu.unsqueeze(1) # Shape: (p, batch_size, d)
            
            M_inv_W_T = torch.bmm(M_inv, W_T)
            temp = torch.bmm(deviation, W)
            temp = torch.bmm(temp, M_inv_W_T)
            
            # Mahalanobis-like distance term
            quadratic_form = (torch.sum(deviation**2, dim=-1) - torch.sum(deviation * temp, dim=-1)) / sigma2.view(p, 1)

            # Calculate log responsibility (unnormalized)
            logR[batch_start:batch_end, :] = (
                torch.log(pi + epsilon) # Add epsilon to prevent log(0)
                + log_det_C_inv_half
                - 0.5 * quadratic_form.T
            )

        # --- Log-likelihood Calculation (for monitoring convergence) ---
        myMax = torch.max(logR, axis=1, keepdim=True).values
        # Handle cases where logR might become -inf
        myMax = torch.nan_to_num(myMax, neginf=-1e30)

        logR_stable = logR - myMax
        log_sum_exp = torch.log(torch.exp(logR_stable).sum(axis=1) + epsilon)
        L[i] = (myMax.squeeze() + log_sum_exp).sum()
        L[i] -= N * d * math.log(2 * math.pi) / 2.0
        
        pbar.set_postfix(log_likelihood=f"{L[i].item():.4f}")

        # Normalize logR to get R (posterior probabilities) using log-sum-exp trick
        log_sum_R = torch.logsumexp(logR, dim=1, keepdim=True)
        logR = logR - log_sum_R
        R = torch.exp(logR)

        # Check for NaN/inf in responsibilities, which indicates a fatal error
        if not torch.all(torch.isfinite(R)):
            print(f"\nFATAL: Responsibilities contain NaN/inf at iteration {i}. Stopping.")
            L[i:] = float('nan')
            break
            
        # --- M-step (Maximization) ---
        R_sum = R.sum(axis=0)
        R_sum_stable = R_sum + (10 * epsilon)

        # Update pi (mixing proportions)
        pi = R_sum / N

        # Update mu (cluster means)
        mu = torch.einsum('np,nd->pd', R, X) / R_sum_stable.unsqueeze(1)

        # Update W and sigma2 (factor loadings and noise variance)
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
            inner_term = (sigma2.view(p, 1, 1) * I_q) + torch.bmm(torch.bmm(M_inv, W_T), S_W)
            inner_inv = torch.linalg.inv(inner_term)
        except torch.linalg.LinAlgError:
            # Regularize if inner term is singular
            inner_term_reg = inner_term + torch.eye(q, device=device, dtype=X.dtype) * 1e-6
            inner_inv = torch.linalg.inv(inner_term_reg)
            
        W_new = torch.bmm(S_W, inner_inv)

        trace_term = torch.einsum('pdq,pqi->pdi', S_W, M_inv)
        trace_term = torch.einsum('pji,pij->p', W_new.transpose(-2,-1), trace_term)

        sigma2_new = (1 / d) * ( (term2_numerator / R_sum_stable) - trace_term )
        
        W = W_new
        sigma2 = sigma2_new

        # --- ▼▼▼【重要】クラスター崩壊防止と再初期化 ▼▼▼ ---
        # If a cluster's responsibility (pi) becomes too small, it has "collapsed".
        # We re-initialize it to give it a chance to capture a new part of the data.
        pi_floor = 1.0 / (N * 10) # Define a minimum responsibility threshold
        dead_clusters_mask = (pi < pi_floor)

        if torch.any(dead_clusters_mask):
            # Find the average variance of healthy clusters for a sensible initialization
            healthy_sigma2_mean = sigma2[~dead_clusters_mask].mean()
            if not torch.isfinite(healthy_sigma2_mean):
                healthy_sigma2_mean = torch.var(X) # Fallback if all clusters are dead

            for c_idx in torch.where(dead_clusters_mask)[0]:
                print(f"\nINFO: Re-initializing collapsed cluster {c_idx.item()} at iteration {i}.")
                # 1. Re-initialize mu to a random data point
                random_idx = torch.randint(0, N, (1,)).item()
                mu[c_idx] = X[random_idx]
                
                # 2. Re-initialize W with small random values
                W[c_idx] = torch.randn_like(W[c_idx]) * 0.1
                
                # 3. Reset the noise variance to a sensible value
                sigma2[c_idx] = healthy_sigma2_mean
                
                # 4. Give the cluster a small amount of "life" by taking from the richest cluster
                if torch.any(~dead_clusters_mask):
                    richest_cluster_idx = torch.argmax(pi)
                    stolen_pi = min(pi[richest_cluster_idx] * 0.1, 0.01) # Steal up to 1%
                    pi[c_idx] += stolen_pi
                    pi[richest_cluster_idx] -= stolen_pi

            # Re-normalize pi after adjustment
            pi = torch.clamp(pi, min=epsilon)
            pi /= pi.sum()
        # --- ▲▲▲ 修正ここまで ▲▲▲ ---

        # Finally, ensure sigma2 does not become negative or zero
        sigma2 = torch.clamp(sigma2, min=epsilon)

    pbar.close()
    
    # Final check for any remaining NaNs in results
    if not torch.all(torch.isfinite(L)):
        print("\nWarning: Log-likelihood contains NaN values after training.")
        
    print("\nDone.")
    return pi, mu, W, sigma2, R, L, sigma2hist