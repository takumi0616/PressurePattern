# Translation in python of the Matlab implementation of Mathieu Andreux and
# Michel Blancard, of the algorithm described in
# "Mixtures of Probabilistic Principal Component Analysers",
# Michael E. Tipping and Christopher M. Bishop, Neural Computation 11(2),
# pp 443–482, MIT Press, 1999

import numpy as np
import numba

# Numbaデコレータを追加して関数をJITコンパイルする
# 注意: variance_levelを使用する場合、Noneではなく数値を渡してください
@numba.jit(nopython=True)
def initialization_kmeans(X, p, q, variance_level):
    """
    X : dataset
    p : number of clusters
    q : dimension of the latent space
    variance_level

    pi : proportions of clusters
    mu : centers of the clusters in the observation space
    W : latent to observation matricies
    sigma2 : noise
    """

    N, d = X.shape

    # initialization
    # Numbaはnp.random.seedをサポートしていないため、外部で設定が必要
    init_centers = np.random.randint(0, N, p)
    # uniqueな中心が選ばれるまで繰り返すロジックは同じ
    while (len(np.unique(init_centers)) != p):
        init_centers = np.random.randint(0, N, p)

    mu = X[init_centers, :]
    distance_square = np.zeros((N, p))
    clusters = np.zeros(N, dtype=np.int32)

    D_old = -2.0
    D = -1.0

    while(D_old != D):
        D_old = D

        # assign clusters
        for c in range(p):
            # distance_square[:, c] = np.power(X - mu[c, :], 2).sum(1)
            # 上記の処理をNumbaが最適化しやすいように明示的なループで記述
            for i in range(N):
                sum_sq = 0.0
                for j in range(d):
                    diff = X[i, j] - mu[c, j]
                    sum_sq += diff * diff
                distance_square[i, c] = sum_sq

        for i in range(N):
            clusters[i] = np.argmin(distance_square[i,:])


        # compute distortion
        dist_sum = 0.0
        for i in range(N):
            dist_sum += distance_square[i, clusters[i]]
        D = dist_sum
        
        distmin = np.zeros(N)
        for i in range(N):
            distmin[i] = distance_square[i, clusters[i]]

        # compute new centers
        # 元のコード `mu[c, :] = X[clusters == c, :].mean(0)` をNumbaフレンドリーに書き換え
        sum_mu = np.zeros((p, d))
        counts = np.zeros(p)
        for i in range(N):
            c_idx = clusters[i]
            sum_mu[c_idx, :] += X[i, :]
            counts[c_idx] += 1
        
        for c in range(p):
            if counts[c] > 0:
                mu[c, :] = sum_mu[c, :] / counts[c]
            # 空のクラスタのケースは元のコードも未定義なため、ここでは何もしない

    # parameter initialization
    pi = np.zeros(p)
    W = np.zeros((p, d, q))
    sigma2 = np.zeros(p)

    # 元の初期化ループをNumbaフレンドリーに書き換え
    # まずクラスタごとの統計量を計算
    cluster_counts = np.zeros(p)
    distmin_sum = np.zeros(p)
    for i in range(N):
        c_idx = clusters[i]
        cluster_counts[c_idx] += 1
        distmin_sum[c_idx] += distmin[i]

    for c in range(p):
        if variance_level != -1.0: # variance_levelが指定されているか（Noneの代わりに-1.0等で判定）
            W[c, :, :] = variance_level * np.random.randn(d, q)
            sigma2[c] = np.abs((variance_level/10) * np.random.randn())
        else:
            W[c, :, :] = np.random.randn(d, q)
            if cluster_counts[c] > 0:
                sigma2[c] = (distmin_sum[c] / cluster_counts[c]) / d
            else:
                sigma2[c] = 1.0 # 空クラスタの場合のデフォルト値

        pi[c] = cluster_counts[c] / N

    return pi, mu, W, sigma2, clusters


# Numbaデコレータを追加. fastmath=Trueでさらに高速化
@numba.jit(nopython=True, fastmath=True)
def mppca_gem(X, pi, mu, W, sigma2, niter):
    N, d = X.shape
    p = len(sigma2)
    _, q = W[0].shape

    sigma2hist = np.zeros((p, niter))
    M = np.zeros((p, q, q))
    Minv = np.zeros((p, q, q))
    # Cinvはサイズが大きいためループ内で生成
    logR = np.zeros((N, p))
    R = np.zeros((N, p))

    L = np.zeros(niter)
    # print文はNumbaのnopythonモードではサポートされないため削除
    for i in range(niter):
        for c in range(p):
            sigma2hist[c, i] = sigma2[c]

            # M
            M_c = sigma2[c]*np.eye(q) + W[c, :, :].T @ W[c, :, :]
            M[c, :, :] = M_c
            Minv[c, :, :] = np.linalg.inv(M_c)

            # Cinv
            W_c = W[c, :, :]
            Minv_c = Minv[c, :, :]
            Cinv_c = (np.eye(d) - W_c @ Minv_c @ W_c.T) / sigma2[c]
            
            # R_ni
            # 改善点: np.log(np.linalg.det) を np.linalg.slogdet に変更
            inner_mat = np.eye(d) - W_c @ Minv_c @ W_c.T
            sign, logdet = np.linalg.slogdet(inner_mat)
            log_det_term = logdet # signは+1と仮定

            deviation_from_center = X - mu[c, :]
            
            # (dev * (dev @ Cinv.T)).sum(1) を効率的に計算
            quad_term = np.zeros(N)
            for k in range(N):
                quad_term[k] = deviation_from_center[k,:] @ Cinv_c @ deviation_from_center[k,:].T

            logR[:, c] = ( np.log(pi[c])
                + 0.5 * log_det_term
                - 0.5 * d * np.log(sigma2[c])
                - 0.5 * quad_term
                )

        # myMaxの計算とlog-sum-expトリック
        myMax = np.zeros(N)
        for k in range(N):
            myMax[k] = np.max(logR[k, :])
        
        log_sum_exp = np.zeros(N)
        for k in range(N):
            log_sum_exp[k] = myMax[k] + np.log(np.sum(np.exp(logR[k, :] - myMax[k])))
        
        L[i] = np.sum(log_sum_exp) - N*d*np.log(2*3.141593)/2.
        
        # logRの正規化
        for k in range(N):
            logR[k, :] = logR[k, :] - log_sum_exp[k]
        
        # piの更新
        log_pi_sum_exp = np.zeros(p)
        myMax_pi = np.zeros(p)
        for c in range(p):
            myMax_pi[c] = np.max(logR[:, c])

        for c in range(p):
            sum_val = 0.0
            for k in range(N):
                sum_val += np.exp(logR[k, c] - myMax_pi[c])
            log_pi_sum_exp[c] = np.log(sum_val)

        logpi = myMax_pi + log_pi_sum_exp - np.log(N)
        pi = np.exp(logpi)
        R = np.exp(logR)
        
        for c in range(p):
            R_c_sum = np.sum(R[:, c])
            mu[c, :] = (R[:, c].reshape(N, 1) * X).sum(axis=0) / R_c_sum
            
            deviation_from_center = X - mu[c, :].reshape(1, d)
            
            SW_numerator = (R[:, c].reshape(N, 1) * deviation_from_center).T @ (deviation_from_center @ W[c,:,:])
            SW = (1 / (pi[c]*N)) * SW_numerator
            
            Wnew = SW @ np.linalg.inv(sigma2[c]*np.eye(q) + Minv[c, :, :] @ W[c, :, :].T @ SW)
            
            term1_num = np.sum(R[:, c].reshape(N, 1) * np.power(deviation_from_center, 2))
            sigma2[c] = (1/d) * ( term1_num / (N*pi[c]) - np.trace(SW @ Minv[c, :, :] @ Wnew.T))

            W[c, :, :] = Wnew

    return pi, mu, W, sigma2, R, L, sigma2hist


# Numbaデコレータを追加
@numba.jit(nopython=True, fastmath=True)
def mppca_predict(X, pi, mu, W, sigma2):
    N, d = X.shape
    p = len(sigma2)
    _, q = W[0].shape

    M = np.zeros((p, q, q))
    Minv = np.zeros((p, q, q))
    logR = np.zeros((N, p))
    R = np.zeros((N, p))

    for c in range(p):
        # M
        M_c = sigma2[c] * np.eye(q) + W[c, :, :].T @ W[c, :, :]
        Minv_c = np.linalg.inv(M_c)

        # Cinv
        W_c = W[c, :, :]
        Cinv_c = (np.eye(d) - W_c @ Minv_c @ W_c.T) / sigma2[c]

        # R_ni
        # 改善点: np.log(np.linalg.det) を np.linalg.slogdet に変更
        inner_mat = np.eye(d) - W_c @ Minv_c @ W_c.T
        sign, logdet = np.linalg.slogdet(inner_mat)
        log_det_term = logdet # signは+1と仮定
        
        deviation_from_center = X - mu[c, :]
        
        quad_term = np.zeros(N)
        for k in range(N):
            quad_term[k] = deviation_from_center[k,:] @ Cinv_c @ deviation_from_center[k,:].T

        logR[:, c] = ( np.log(pi[c])
            + 0.5 * log_det_term
            - 0.5*d*np.log(sigma2[c])
            - 0.5 * quad_term
            )

    # log-sum-expトリックによる正規化
    myMax = np.zeros(N)
    for k in range(N):
        myMax[k] = np.max(logR[k, :])

    log_sum_exp = np.zeros(N)
    for k in range(N):
        log_sum_exp[k] = myMax[k] + np.log(np.sum(np.exp(logR[k, :] - myMax[k])))

    for k in range(N):
        logR[k, :] = logR[k, :] - log_sum_exp[k]
        
    R = np.exp(logR)

    return R