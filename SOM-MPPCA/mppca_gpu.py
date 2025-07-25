# src/PressurePattern/SOM-MPPCA/mppca_gpu.py

import cupy as cp
from tqdm import tqdm # tqdmをインポート

def initialization_kmeans_gpu(X, p, q, variance_level):
    """
    GPU (CuPy) を使用してK-means法による初期化を行う。

    X : cupy.ndarray - GPU上のデータセット
    p : int - クラスタ数
    q : int - 潜在空間の次元数
    variance_level: float - 分散のレベル

    戻り値:
    pi, mu, W, sigma2, clusters - すべてcupy.ndarray
    """
    N, d = X.shape

    # 初期化
    # CuPyの乱数生成器を使用
    init_centers = cp.random.randint(0, N, p)
    # uniqueな中心が選ばれるまで繰り返す
    while (len(cp.unique(init_centers)) != p):
        init_centers = cp.random.randint(0, N, p)

    mu = X[init_centers, :]
    clusters = cp.zeros(N, dtype=cp.int32)
    
    D_old = cp.array([-2.0]) # Dと比較するためにCuPy配列に
    D = cp.array([-1.0])

    # K-meansの反復回数に上限を設ける（無限ループ防止）
    for _ in range(100): 
        # 各データ点と各クラスタ中心との距離を計算 (ブロードキャストを利用)
        # (N, 1, d) - (1, p, d) -> (N, p, d) -> (N, p)
        distance_square = cp.power(X[:, None, :] - mu[None, :, :], 2).sum(axis=2)

        # 各データ点が属するクラスタを決定
        clusters = cp.argmin(distance_square, axis=1)

        # 歪み（distortion）を計算
        D_new = cp.sum(distance_square[cp.arange(N), clusters])

        # 収束判定
        if cp.isclose(D_old, D_new):
            break
        D_old = D_new

        # 新しいクラスタ中心を計算
        for c in range(p):
            mask = (clusters == c)
            if cp.any(mask):
                mu[c, :] = X[mask].mean(axis=0)
            else:
                # クラスタが空になった場合、最も遠い点を新しい中心とする
                # この処理は省略しても良いが、安定性のため
                dists = distance_square[cp.arange(N), clusters]
                farthest_point_idx = cp.argmax(dists)
                mu[c,:] = X[farthest_point_idx]


    # パラメータの初期化
    pi = cp.zeros(p)
    W = cp.zeros((p, d, q))
    sigma2 = cp.zeros(p)

    # クラスタごとの統計量を計算
    cluster_counts = cp.bincount(clusters, minlength=p)
    distmin = distance_square[cp.arange(N), clusters]

    for c in range(p):
        if variance_level != -1.0:
            W[c, :, :] = variance_level * cp.random.randn(d, q, dtype=X.dtype)
            sigma2[c] = cp.abs((variance_level/10) * cp.random.randn(dtype=X.dtype))
        else:
            W[c, :, :] = cp.random.randn(d, q, dtype=X.dtype)
            if cluster_counts[c] > 0:
                sigma2[c] = cp.sum(distmin[clusters == c]) / (cluster_counts[c] * d)
            else:
                sigma2[c] = 1.0

        pi[c] = cluster_counts[c] / N

    return pi, mu, W, sigma2, clusters


def mppca_gem_gpu(X, pi, mu, W, sigma2, niter):
    """
    GPU (CuPy) を使用してMPPCAのGEMアルゴリズムを実行する。
    進捗表示のためにtqdmを統合。

    X, pi, mu, W, sigma2: cupy.ndarray - GPU上の初期パラメータ

    戻り値:
    pi, mu, W, sigma2, R, L, sigma2hist - すべてcupy.ndarray
    """
    N, d = X.shape
    p = len(sigma2)
    _, q = W[0].shape

    sigma2hist = cp.zeros((p, niter))
    L = cp.zeros(niter)

    # eye（単位行列）を事前に作成
    eye_q = cp.eye(q, dtype=X.dtype)
    eye_d = cp.eye(d, dtype=X.dtype)

    # ★★★ 改善点: tqdmをループに適用 ★★★
    iterator = tqdm(range(niter), desc="MPPCA Training (GPU)")
    for i in iterator:
        sigma2hist[:, i] = sigma2

        # --- E-Step ---
        M = sigma2[:, None, None] * eye_q + W.transpose(0, 2, 1) @ W
        Minv = cp.linalg.inv(M)
        
        log_det_C = (d - q) * cp.log(sigma2) + cp.linalg.slogdet(M)[1]

        X_minus_mu = X[:, None, :] - mu[None, :, :]
        
        W_Minv = W @ Minv
        W_Minv_WT = W_Minv @ W.transpose(0, 2, 1)
        
        X_minus_mu_p = X_minus_mu.transpose(1,0,2)
        mahalanobis_term = cp.einsum('pni,pij,pnj->pn', X_minus_mu_p, (eye_d - W_Minv_WT), X_minus_mu_p) / sigma2[:,None]

        logR = (cp.log(pi) - 0.5 * (d * cp.log(2 * cp.pi) + log_det_C + mahalanobis_term.T))
        
        myMax = cp.max(logR, axis=1, keepdims=True)
        log_sum_exp = myMax + cp.log(cp.sum(cp.exp(logR - myMax), axis=1, keepdims=True))
        
        L[i] = cp.sum(log_sum_exp)
        
        R = cp.exp(logR - log_sum_exp)

        # --- M-Step ---
        R_sum = cp.sum(R, axis=0)
        
        # ゼロ除算を避けるための微小値
        R_sum = cp.where(R_sum == 0, 1e-9, R_sum)
        
        pi = R_sum / N
        
        mu = (R.T @ X) / R_sum[:, None]
        
        X_minus_mu = X[:, None, :] - mu[None, :, :]

        S = X_minus_mu.transpose(1,0,2) * R.T[:, :, None]
        S = cp.einsum('pni,pnj->pij', S, X_minus_mu.transpose(1,0,2)) / R_sum[:, None, None]

        try:
            inv_term = cp.linalg.inv(sigma2[:, None, None] * eye_q + Minv @ W.transpose(0, 2, 1) @ S)
            W_new = S @ W @ inv_term
        except cp.linalg.LinAlgError:
            # 稀に発生する特異行列エラーへの対処
            tqdm.write(f"Warning: Singular matrix in W_new update at iteration {i}. Using pseudo-inverse.")
            inv_term = cp.linalg.pinv(sigma2[:, None, None] * eye_q + Minv @ W.transpose(0, 2, 1) @ S)
            W_new = S @ W @ inv_term

        trace_term = cp.trace(W_new.transpose(0, 2, 1) @ S @ W @ Minv, axis1=1, axis2=2)
        sigma2_new = (1/d) * (cp.trace(S, axis1=1, axis2=2) - trace_term)

        # sigma2が負または非常に小さくなるのを防ぐ
        sigma2 = cp.maximum(sigma2_new, 1e-9)

        W = W_new
        
        # tqdmの進捗バーに現在の対数尤度を表示
        iterator.set_postfix(log_likelihood=f"{L[i].item():.2f}")


    return pi, mu, W, sigma2, R, L, sigma2hist