import numpy as np

class ACS:
    """
    Adaptive Competitive Self-organizing (ACS) モデルの実装。
    論文: "Real-time classifier based on adaptive competitive self-organizing algorithm"
    Zahra Sarafraz, Hossein Sarafraz and Mohammad R Sayeh (2018)
    Adaptive Behavior, Vol. 26(1) 21-31
    DOI: 10.1177/1059712318760695
    本実装では、論文の記述に基づきクラスタ数の動的変更機能（新規生成・削除）を目指しています。
    論文の主な特徴である動的構造と自己調整パラメータ [cite: 3]、寄生的限界点問題への対処 [cite: 3] を実装します。
    """

    def __init__(self,
                 gamma,
                 beta,
                 learning_rate_W,
                 learning_rate_lambda,
                 learning_rate_Z,
                 max_clusters=10,
                 initial_clusters=None, # fit時に初期生成するクラスタ数。Noneまたは0なら最初のデータで1つ。
                 n_features=None,       # fit時にデータから推定可能。
                 activation_type='elliptical',
                 initial_lambda_scalar=0.1,     # 円形の場合のlambda_jの初期値
                 initial_lambda_vector_val=0.1, # 楕円の場合のlambda_ijの初期値
                 initial_lambda_crossterm_val=0.01, # 楕円の場合のlambda_Kjの初期値
                 initial_Z_val=0.5,             # _create_initial_set_of_clusters で初期化されるクラスタのZの初期値
                 initial_Z_new_cluster=0.2,     # ★変更: 新規生成クラスタのZの初期値 (低めに設定し、データで育つかを見る)
                 lambda_min_val=1e-6,           # lambda_ij の最小値 (クリッピング用、数値的安定性のため)
                 bounds_W=None,                 # ラベル W_ij の値域制限タプル (min, max)
                 theta_new=0.1,                 # 新規クラスタを生成する活性化の閾値
                 death_patience_steps=1000,     # クラスタが削除されるまでの非活性許容ステップ数
                 Z_death_threshold=0.05,        # クラスタ削除の判断に使われるZ値の閾値
                 random_state=None):
        """
        ACSクラスの初期化 (動的クラスタ数変更機能付き)

        Args:
            gamma (float): エネルギー関数の目標値に関連するパラメータ [cite: 54]。
            beta (float): 競争係数 [cite: 75]。
            learning_rate_W (float): ラベル W_ij の学習率。
            learning_rate_lambda (float): 警戒パラメータ lambda の学習率。
            learning_rate_Z (float): 深さパラメータ Z_j の学習率。
            max_clusters (int): 許容される最大のクラスタ数。論文では "maximum possible number of the labels" [cite: 46] と言及。
            initial_clusters (int, optional): 最初に生成するクラスタ数。Noneまたは0の場合、最初のデータ入力で1つ生成。1以上の場合、fit時にその数だけ初期化。
            n_features (int, optional): N, 入力データの特徴次元数。Noneの場合、fit時にデータから推定。
            activation_type (str): 活性化関数のタイプ ('circular' または 'elliptical')。
                                   'circular': 式(2)ベース [cite: 54]。
                                   'elliptical': 式(6)ベース [cite: 123]。
            initial_lambda_scalar (float): 円形活性化の場合の警戒パラメータ lambda_j の初期値。
            initial_lambda_vector_val (float): 楕円活性化の場合の次元ごとの警戒パラメータ lambda_ij の初期値。
            initial_lambda_crossterm_val (float): 楕円活性化の場合のクロスターム警戒パラメータ lambda_Kj の初期値。
            initial_Z_val (float): _create_initial_set_of_clustersで初期化されるクラスタのZの初期値。
            initial_Z_new_cluster (float): _add_new_clusterで新規生成されるクラスタのZの初期値。
            lambda_min_val (float): 警戒パラメータ lambda_ij が取りうる最小値 (クリッピング用、論文外の処理)。
            bounds_W (tuple, optional): ラベル W_ij の値を制限する範囲 (min, max)。敗者ラベル問題への対応 [cite: 73, 74]。
            theta_new (float): 新規クラスタを生成するかどうかの判断に使われる活性化値の閾値。
                                 最も活性の高い既存クラスタの活性化値がこの閾値より低い場合に新規クラスタ生成を検討。
                                 論文の "if the new input pattern does not resemble any formed clusters, then a new cluster will be generated" [cite: 192] に基づく。
            death_patience_steps (int): クラスタが削除されるまでの非活性許容ステップ数。
                                       このステップ数を超えて勝利せず、かつZ値が低い場合に削除。
                                       論文の「寄生的アトラクタの誘引域が弱まり最終的に消滅する」("basins of attraction for parasitic attractors becomes weaker and eventually disappears") [cite: 132] 概念に基づく。
            Z_death_threshold (float): クラスタ削除の判断に使われるZ値の閾値。この値よりZが低いと削除候補。
            random_state (int, optional): 乱数生成器のシード値。
        """
        self.gamma = gamma
        self.beta = beta
        self.eta_W = learning_rate_W
        self.eta_lambda = learning_rate_lambda
        self.eta_Z = learning_rate_Z
        self.N = n_features # fit時に確定
        self.activation_type = activation_type.lower()
        if self.activation_type not in ['circular', 'elliptical']:
            raise ValueError("activation_type は 'circular' または 'elliptical' である必要があります。")

        self.max_clusters = max_clusters
        self.initial_clusters_to_create = initial_clusters if initial_clusters is not None else 0

        self.initial_lambda_scalar = initial_lambda_scalar
        self.initial_lambda_vector_val = initial_lambda_vector_val
        self.initial_lambda_crossterm_val = initial_lambda_crossterm_val
        self.initial_Z_val = initial_Z_val
        self.initial_Z_new_cluster = initial_Z_new_cluster

        self.lambda_min_val = lambda_min_val
        self.bounds_W = bounds_W

        self.theta_new = theta_new
        self.death_patience_steps = death_patience_steps
        self.Z_death_threshold = Z_death_threshold

        if random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random.RandomState()

        # パラメータ配列 (Nが確定してから正しい次元で初期化)
        self.W = np.empty((0, 0), dtype=float) # (M x N)
        self.lambdas = np.empty((0, 0), dtype=float) # (M x 1 or M x (N+1))
        self.Z = np.empty((0,1), dtype=float)      # (M x 1)
        self.inactive_steps = np.empty((0,1), dtype=int) # (M x 1)
        self.M = 0          # 現在のクラスタ数

        self.fitted_ = False # モデルが有効なクラスタを持っているか
        self._first_data_processed_for_N = False # Nを決定するために最初のデータが処理されたか

    def _initialize_all_parameters_arrays(self, n_features_from_data):
        """
        特徴次元数 N が確定した後に、各パラメータ配列を正しい次元で(空の状態で)初期化する。
        """
        if self.N is not None and self.N != n_features_from_data:
             raise ValueError(f"入力データの特徴次元数 ({n_features_from_data}) が初期化時の n_features ({self.N}) と一致しません。")
        self.N = n_features_from_data
        
        self.W = np.empty((0, self.N), dtype=float)
        if self.activation_type == 'circular':
            self.lambdas = np.empty((0, 1), dtype=float)
        else:
            if self.N is None: # Should not happen if n_features_from_data is passed
                raise ValueError("Elliptical activation requires N to be defined.")
            self.lambdas = np.empty((0, self.N + 1), dtype=float)
        self.Z = np.empty((0,1), dtype=float)
        self.inactive_steps = np.empty((0,1), dtype=int)
        self.M = 0
        self._first_data_processed_for_N = True

    def _create_initial_set_of_clusters(self, X_train_sample_for_init):
        """
        fit時に initial_clusters_to_create の数だけクラスタを初期生成する。
        X_train_sample_for_init は初期化に使用するデータサンプル。
        """
        if self.initial_clusters_to_create <= 0 or not self._first_data_processed_for_N:
            return

        num_to_initialize = min(self.initial_clusters_to_create, self.max_clusters)
        if num_to_initialize == 0:
            return

        # print(f"初期クラスタ群を {num_to_initialize} 個生成します。")

        num_available_samples = X_train_sample_for_init.shape[0]
        if num_available_samples == 0:
            # print("警告: 初期クラスタ生成のためのサンプルデータがありません。")
            return
            
        if num_available_samples < num_to_initialize:
            # print(f"警告: 初期化に使用するサンプル数 ({num_available_samples}) が要求クラスタ数 ({num_to_initialize}) より少ないため、ラベルの初期値は重複して選択される可能性があります。")
            indices = self.rng.choice(num_available_samples, num_to_initialize, replace=True)
        else:
            indices = self.rng.choice(num_available_samples, num_to_initialize, replace=False)
        
        # W_ij の初期化: 学習データからランダムにM個のサンプルを選択 [cite: 52]
        self.W = X_train_sample_for_init[indices].copy()

        if self.activation_type == 'circular':
            self.lambdas = np.full((num_to_initialize, 1), self.initial_lambda_scalar, dtype=float)
        else: # elliptical
            _lambdas_ij = np.full((num_to_initialize, self.N), self.initial_lambda_vector_val, dtype=float)
            _lambdas_Kj = np.full((num_to_initialize, 1), self.initial_lambda_crossterm_val, dtype=float)
            self.lambdas = np.hstack((_lambdas_ij, _lambdas_Kj))
        
        # Z_j の初期化: Z_j in [0,1] [cite: 133]。
        self.Z = np.full((num_to_initialize, 1), self.initial_Z_val, dtype=float)
        self.inactive_steps = np.zeros((num_to_initialize, 1), dtype=int)
        self.M = num_to_initialize
        
        if self.M > 0:
            self.fitted_ = True # クラスタが初期化されたことを示す

    def _add_new_cluster(self, U_p_row):
        """
        新しいクラスタを入力パターン U_p_row で初期化し、パラメータ配列に追加する。
        論文の「新しい入力パターンが既存のどのクラスタとも十分に類似していない場合、新しいクラスタが生成される」 [cite: 192] 概念に基づく。
        """
        if self.M >= self.max_clusters:
            return False

        new_W_row = U_p_row.reshape(1, -1)
        self.W = np.vstack((self.W, new_W_row))

        if self.activation_type == 'circular':
            new_lambda_row = np.full((1, 1), self.initial_lambda_scalar, dtype=float)
        else: # elliptical
            _lambda_ij_row = np.full((1, self.N), self.initial_lambda_vector_val, dtype=float)
            _lambda_Kj_row = np.full((1, 1), self.initial_lambda_crossterm_val, dtype=float)
            new_lambda_row = np.hstack((_lambda_ij_row, _lambda_Kj_row))
        self.lambdas = np.vstack((self.lambdas, new_lambda_row))

        # 新規クラスタのZ初期値 [cite: 133]。論文外の工夫として低めに設定し、データで育つかを見る。
        new_Z_row = np.full((1,1), self.initial_Z_new_cluster, dtype=float)
        self.Z = np.vstack((self.Z, new_Z_row))
        
        new_inactive_steps_row = np.zeros((1,1), dtype=int) # 新規クラスタは非活性0からスタート
        self.inactive_steps = np.vstack((self.inactive_steps, new_inactive_steps_row))
        
        self.M += 1
        if not self.fitted_ and self.M > 0 : # 最初のクラスタが生成された場合
            self.fitted_ = True
        return True

    def _delete_cluster(self, cluster_index_to_delete):
        """
        指定されたインデックスのクラスタを削除する。
        論文の「寄生的アトラクタの誘引域が弱まり最終的に消滅する」("basins of attraction for parasitic attractors becomes weaker and eventually disappears") [cite: 132] 概念に基づく。
        """
        if not (0 <= cluster_index_to_delete < self.M):
            return

        self.W = np.delete(self.W, cluster_index_to_delete, axis=0)
        self.lambdas = np.delete(self.lambdas, cluster_index_to_delete, axis=0)
        self.Z = np.delete(self.Z, cluster_index_to_delete, axis=0)
        self.inactive_steps = np.delete(self.inactive_steps, cluster_index_to_delete, axis=0)
        self.M -= 1
        if self.M == 0: # 全てのクラスタが削除された場合
            self.fitted_ = False

    def _compute_activation_circular(self, U_p_row, W_j_row, lambda_j_scalar):
        """ 円形の活性化関数 X_jp を計算する。論文 式(2) [cite: 54]。 """
        diff_sq_norm = np.sum((U_p_row - W_j_row)**2) # ユークリッド距離の二乗 [cite: 56]
        denominator = 1.0 + lambda_j_scalar * diff_sq_norm
        if denominator < 1e-9: return 1.0 / 1e-9
        return 1.0 / denominator

    def _compute_activation_elliptical(self, U_p_row, W_j_row, lambda_ij_vector, lambda_Kj_scalar):
        """ 楕円形の活性化関数 X_jp を計算する。論文 式(6) [cite: 123]。 """
        diff = U_p_row - W_j_row # (1 x N)
        # sum_{i=1}^N (lambda_ij * (U_ip - W_ij)^2)
        sum_term = np.sum(lambda_ij_vector * (diff**2))
        
        # product_{i=1}^N (U_ip - W_ij) : 論文式(6)の積の項
        if self.N == 0: prod_term = 0.0 # N=0は実質ありえない
        elif self.N == 1: prod_term = diff[0,0]
        else: prod_term = np.prod(diff)
            
        denominator = 1.0 + sum_term + lambda_Kj_scalar * prod_term
        if np.abs(denominator) < 1e-9:
            return (1.0 / 1e-9) if denominator >= 0 else -(1.0/1e-9)
        return 1.0 / denominator

    def _calculate_all_activations(self, U_p_row):
        """ 全ての現存クラスタに対する活性化値 X_p を計算する。 """
        if self.M == 0:
            return np.empty((0,1), dtype=float)

        X_p = np.zeros((self.M, 1), dtype=float)
        for j_idx in range(self.M):
            if self.activation_type == 'circular':
                X_p[j_idx, 0] = self._compute_activation_circular(U_p_row, self.W[j_idx, :], self.lambdas[j_idx, 0])
            else: # elliptical
                X_p[j_idx, 0] = self._compute_activation_elliptical(U_p_row, self.W[j_idx, :], self.lambdas[j_idx, :-1], self.lambdas[j_idx, -1])
        return X_p

    def partial_fit(self, U_p):
        """
        単一の入力パターン U_p でモデルをオンライン学習する。
        クラスタの動的生成・削除機能を含む。
        論文のリアルタイム分類器の概念 [cite: 1] に基づき、逐次的に学習。
        """
        U_p_row = U_p.reshape(1, -1)

        if not self._first_data_processed_for_N: # Nが未確定なら、現在の入力で確定し配列準備
            self._initialize_all_parameters_arrays(U_p_row.shape[1])

        # ---- 1. 最初のクラスタ生成 (もしクラスタが存在しない場合) ----
        if self.M == 0:
            if not self._add_new_cluster(U_p_row): # 最初のクラスタを現在の入力で生成
                return # 生成失敗 (max_clusters=0など) の場合は学習スキップ

        # ---- 2. 全クラスタの活性化値計算 ----
        X_p = self._calculate_all_activations(U_p_row)
        
        if self.M == 0: #  _add_new_cluster が失敗した場合など (実質上記でreturnされる)
            return

        max_activation_before_birth = np.max(X_p) if self.M > 0 else 0.0
        
        # ---- 3. 新規クラスタ生成の試行 (Birth) ----
        # 論文 "if the new input pattern does not resemble any formed clusters, then a new cluster will be generated" [cite: 192]
        new_cluster_added_this_step = False
        if max_activation_before_birth < self.theta_new and self.M < self.max_clusters:
            if self._add_new_cluster(U_p_row): # 新規クラスタを現在の入力で生成
                new_cluster_added_this_step = True
                X_p = self._calculate_all_activations(U_p_row) # 活性化値を再計算

        # ---- 4. 勝者クラスタの最終決定 ----
        if self.M == 0: # 再計算後、万一M=0なら終了 (削除ロジックが過敏な場合など)
            return
            
        winner_cluster_index = np.argmax(X_p[:,0]) # 現状のX_pに基づく勝者
        if new_cluster_added_this_step:
             # このステップで新しいクラスタが追加された場合、そのクラスタを今回の勝者とする
            winner_cluster_index = self.M - 1 # 新しく追加されたクラスタは最後尾


        # ---- 5. パラメータ更新 ----
        # エネルギー関数 E (式(9) [cite: 133]) の関連項
        sum_Z_X = np.sum(self.Z * X_p)
        # term_in_bracket_E1_Z: (gamma - sum_q Z_q X_q), 式(10),(11),(7)の主要項 [cite: 136, 126]
        term_in_bracket_E1_Z = self.gamma - sum_Z_X
        # term_in_bracket_E1_noZ: (gamma - sum_q X_q), 式(5)の主要項 [cite: 114]
        term_in_bracket_E1_noZ = self.gamma - np.sum(X_p)

        # 5.a. 深さパラメータ Z_j の更新
        # 論文 式(10): dZ_j/dt = 2 * (gamma - sum_q Z_qp X_qp) * X_jp * Z_j * (Z_j - 1) [cite: 136]
        # この式は Z_j=0 (敗者) または Z_j=1 (勝者) に収束させる効果がある。
        delta_Z = 2 * term_in_bracket_E1_Z * X_p * self.Z * (self.Z - 1)
        self.Z += self.eta_Z * delta_Z
        self.Z = np.clip(self.Z, 0.0, 1.0) # Z_j in [0,1] の制約 [cite: 133]

        # 5.b. ラベル W_ij の更新
        # 論文 式(11)ベース: dW_ij/dt = 4 * bracket_W * X_jp^2 * effective_lambda_term [cite: 136]
        # bracket_W = [Z_j(gamma - sum_q Z_q X_q) - 2*beta*sum_{k!=j}X_k]
        # effective_lambda_term は円形なら (U_ip - W_ij) * lambda_j
        for j_idx in range(self.M):
            sum_X_others = np.sum(X_p) - X_p[j_idx, 0] # 競合項 sum_{k!=j}X_kp
            bracket_term_W = self.Z[j_idx, 0] * term_in_bracket_E1_Z - 2 * self.beta * sum_X_others
            
            diff_U_W_j = U_p_row - self.W[j_idx, :] # (U_p - W_j), (1 x N)

            if self.activation_type == 'circular':
                lambda_j_val = self.lambdas[j_idx, 0] # スカラー lambda_j from 式(11)
                effective_lambda_term_W = lambda_j_val * diff_U_W_j # (1 x N)
            else: # elliptical
                # 楕円形の場合のW更新: 式(11)の構造とdX_jp/dW_kから類推
                # dW_k/dt ∝ X_jp^2 * [ lambda_kj*2(U_k-W_k) + lambda_Kj*prod_{l!=k}(U_l-W_l) ]
                lambda_ij_vec = self.lambdas[j_idx, :-1].flatten() # (N,)
                lambda_Kj_s = self.lambdas[j_idx, -1]    # scalar
                
                effective_lambda_term_W = np.zeros_like(diff_U_W_j) # (1 x N)
                for i_target_dim in range(self.N):
                    # term1: lambda_kj * 2 * (U_k - W_k)
                    term1 = lambda_ij_vec[i_target_dim] * diff_U_W_j[0, i_target_dim] * 2.0
                    
                    # term2: lambda_Kj * product_{l!=k}(U_l - W_l)
                    prod_others_val = 1.0
                    if self.N > 1 :
                        for k_prod_dim in range(self.N):
                            if k_prod_dim != i_target_dim:
                                prod_others_val *= diff_U_W_j[0, k_prod_dim]
                    elif self.N == 1: # N=1の場合、他の項の積は1と解釈 (またはlambda_Kj項は別の形になる)
                        prod_others_val = 1.0 # ここでは便宜上1とするが、厳密にはN=1の時の式(6)の微分を考えるべき
                    
                    term2 = lambda_Kj_s * prod_others_val
                    effective_lambda_term_W[0, i_target_dim] = term1 + term2
            
            # 式(11)の係数4を乗じる (ただし、effective_lambda_term_Wがベクトルなのでスカラーlambda_jの場合と等価かは議論の余地)
            delta_W_j_row = 4 * bracket_term_W * (X_p[j_idx, 0]**2) * effective_lambda_term_W
            self.W[j_idx, :] += (self.eta_W * delta_W_j_row).flatten()

        if self.bounds_W is not None: # W_ij の値域制限 [cite: 73, 74]
            self.W = np.clip(self.W, self.bounds_W[0], self.bounds_W[1])

        # 5.c. 警戒パラメータ Lambda の更新
        for j_idx in range(self.M):
            sum_X_others = np.sum(X_p) - X_p[j_idx, 0] # 競合項
            
            if self.activation_type == 'circular':
                # 論文 式(5): dE/d(lambda_j) = 2 * bracket_lambda * X_jp^2 * ||U-W||^2 [cite: 114]
                # bracket_lambda = [(gamma - sum_q X_q) - beta*sum_{k!=j}X_k]
                # d(lambda)/dt = -eta * dE/d(lambda) (勾配降下)
                bracket_term_lambda_circ = term_in_bracket_E1_noZ - self.beta * sum_X_others
                diff_sq_norm_j = np.sum((U_p_row - self.W[j_idx, :])**2) # ||U_p - W_j||^2
                
                grad_E_lambda_j = 2 * bracket_term_lambda_circ * (X_p[j_idx, 0]**2) * diff_sq_norm_j
                self.lambdas[j_idx, 0] -= self.eta_lambda * grad_E_lambda_j
                self.lambdas[j_idx, 0] = np.maximum(self.lambdas[j_idx, 0], self.lambda_min_val) # lambda_j > 0
            else: # elliptical
                # 論文 式(7): d(lambda_ij)/dt = 2 * bracket_lambda * X_jp^2 * A(i) [cite: 126]
                # bracket_lambda = [(gamma - sum_q Z_q X_q) - beta*sum_{k!=j}X_k]
                # この式は d(lambda)/dt を直接与えているので、符号は論文の式に従う（加算）。
                bracket_term_lambda_ellipt = term_in_bracket_E1_Z - self.beta * sum_X_others
                
                diff_U_W_j_flat = (U_p_row - self.W[j_idx, :]).flatten() # (N,)
                
                # A(i) for lambda_ij (i=1..N) and A(N+1) for lambda_Kj. 論文 式(8) [cite: 126]
                A_terms_dim_lambda = (diff_U_W_j_flat**2) # (U_ip - W_ij)^2, (N,)
                
                if self.N == 0: A_term_cross_lambda = 0.0
                elif self.N == 1: A_term_cross_lambda = diff_U_W_j_flat[0] # product_{k=1}^N (U_kp - W_kj)
                else: A_term_cross_lambda = np.prod(diff_U_W_j_flat)

                # 式(7)の共通係数: 2 * bracket_term * X_jp^2
                update_factor_lambda = 2 * bracket_term_lambda_ellipt * (X_p[j_idx, 0]**2)
                
                # lambda_ij (次元ごとの警戒パラメータ) の更新
                delta_lambda_ij_vec = update_factor_lambda * A_terms_dim_lambda # (N,)
                self.lambdas[j_idx, :-1] += self.eta_lambda * delta_lambda_ij_vec
                
                # lambda_Kj (クロスタームの警戒パラメータ) の更新
                delta_lambda_Kj_scalar = update_factor_lambda * A_term_cross_lambda # scalar
                self.lambdas[j_idx, -1] += self.eta_lambda * delta_lambda_Kj_scalar

                # lambda_ij > 0 の制約 [cite: 125] (論文外の明示的クリッピング)
                self.lambdas[j_idx, :-1] = np.maximum(self.lambdas[j_idx, :-1], self.lambda_min_val)
                # Note: 楕円条件 (e.g., (lambda_3j)^2 - 4(lambda_1j)(lambda_2j) < 0 for 2D) は陽に扱っていない [cite: 125]

        # ---- 6. 非活性ステップの更新と不要クラスタ削除 (Death) ----
        # 論文の「寄生的アトラクタの誘引域が弱まり最終的に消滅する」概念に基づく [cite: 132]。
        if self.M > 0 : # クラスタが存在する場合のみ
            self.inactive_steps += 1 # 全てのクラスタの非活性カウントを増やす
            if winner_cluster_index != -1 and 0 <= winner_cluster_index < self.M : # 有効な勝者がいればリセット
                 self.inactive_steps[winner_cluster_index, 0] = 0

            # 削除候補のインデックスを見つける (少なくとも1つのクラスタは残すように M > 1 の場合のみ削除)
            if self.M > 1:
                indices_to_delete = []
                for j_idx in range(self.M):
                    if self.inactive_steps[j_idx, 0] > self.death_patience_steps and \
                       self.Z[j_idx, 0] < self.Z_death_threshold:
                        indices_to_delete.append(j_idx)
                
                # 見つかったインデックスを降順にソートして削除 (ループ中のインデックスのずれを防ぐため)
                for j_idx_del in sorted(indices_to_delete, reverse=True):
                    self._delete_cluster(j_idx_del)

    def fit(self, X_train, epochs=1):
        """
        学習データ X_train を用いてモデルを学習する。
        オンライン学習をエポック数だけ繰り返す。
        """
        if X_train.shape[0] == 0:
            return

        # Nの初期化とパラメータ配列の準備 (最初のfit呼び出し時のみ)
        if not self._first_data_processed_for_N:
            self._initialize_all_parameters_arrays(X_train.shape[1])

        # initial_clusters_to_create に基づく初期クラスタ群の生成 (最初のfit呼び出し時、かつM=0の場合のみ)
        if self.M == 0 and self.initial_clusters_to_create > 0:
            self._create_initial_set_of_clusters(X_train)

        n_samples = X_train.shape[0]
        for epoch in range(epochs):
            indices = self.rng.permutation(n_samples)
            X_shuffled = X_train[indices]

            for i in range(n_samples):
                U_p = X_shuffled[i, :]
                self.partial_fit(U_p)
        
        # fit後、有効なクラスタが存在すれば fitted_ = True とする
        if self.M > 0:
            self.fitted_ = True
        else:
            self.fitted_ = False


    def predict_cluster_label(self, U_p):
        """
        単一の入力パターン U_p に対するクラスタラベルを予測する。
        最も活性化が高い (X_jp が最大) ラベルを割り当てる。
        この選択基準は、論文中の「最短ユークリッド距離を持つラベルベクタ...が最も調整される」 [cite: 47] や
        「類似の入力パターンの集合から最小距離にあるラベルが勝つ可能性が高い」 [cite: 78] という記述と整合する。
        活性化関数 X_jp は距離が小さいほど大きな値を取るため、最大活性化の選択はこれに対応する。
        """
        if not self.fitted_ or self.M == 0: # 学習済みでなく、有効なクラスタがない場合は予測不可
            return -1 

        U_p_row = U_p.reshape(1, -1)
        X_values = self._calculate_all_activations(U_p_row) # (M x 1)
        
        if self.M == 0: # 再度チェック (e.g., _calculate_all_activations で M が変わりうるような極端なケースはないはずだが)
            return -1
            
        # 全ての活性化値が0 (または非常に小さい) 場合、どのクラスタにも属さないと判断することもできる。
        # ここでは、最も活性化が高いものを返す。
        if np.all(X_values <= 1e-9) and X_values.size > 0 : # 論文外の閾値処理: 全ての活性がほぼゼロなら未分類
            return -1 
        return np.argmax(X_values[:,0]) # 最も活性化の高いクラスタのインデックス

    def predict(self, X_test):
        """
        テストデータ X_test に対するクラスタラベルを予測する。
        """
        if not self.fitted_ or self.M == 0 :
             raise RuntimeError("モデルがまだ学習されていません（有効なクラスタが存在しません）。先に fit() メソッドを呼び出してください。")
        
        n_samples = X_test.shape[0]
        labels = np.full(n_samples, -1, dtype=int) # 未分類を-1として初期化
        for i in range(n_samples):
            labels[i] = self.predict_cluster_label(X_test[i, :])
        return labels

    def get_cluster_centers(self):
        """
        学習済みのクラスタ中心 (ラベル W_ij) を返す。
        """
        if not self.fitted_ or self.M == 0:
            # 有効なクラスタがない場合は空の配列を返す
            return np.empty((0, self.N if self.N is not None else 0)) 
        return self.W.copy()
    
    def calculate_energy_at_point(self, U_p_point_original_space):
        """
        指定された単一のデータポイント (元の特徴空間) における
        エネルギー関数 E の値を計算する。論文 式(9) [cite: 133]。
        E = [gamma - sum_{j=1}^M Z_j X_j]^2 + beta * sum_{s=1}^M sum_{j != s}^M X_s X_j

        Args:
            U_p_point_original_space (np.ndarray): (N,) or (1, N) 形状のデータポイント。

        Returns:
            float: 計算されたエネルギー値。
        """
        if not self.fitted_ or self.M == 0:
            # print("警告: エネルギー計算時に有効なクラスタが存在しません。大きな値を返します。")
            return np.inf # または適切なデフォルト値

        U_p_row = U_p_point_original_space.reshape(1, -1)
        if U_p_row.shape[1] != self.N:
            raise ValueError(f"入力ポイントの次元 ({U_p_row.shape[1]}) がモデルの特徴次元 ({self.N}) と一致しません。")

        X_values = self._calculate_all_activations(U_p_row) # (M x 1) [cite: 54, 123]

        # 第1項: [gamma - sum_{j=1}^M Z_j X_j]^2 [cite: 133]
        sum_Z_X = np.sum(self.Z * X_values)
        first_term = (self.gamma - sum_Z_X)**2

        # 第2項 (競合項): beta * sum_{s=1}^M sum_{j != s}^M X_s X_j [cite: 133, 72]
        # これは beta * sum_s (X_s * (sum_all_X - X_s)) と等価
        competition_term_value = 0.0
        if self.M > 1: # クラスタが2つ以上の場合のみ競合が発生
            sum_all_X_scalar = np.sum(X_values[:,0]) # (M,) -> scalar
            for s_idx in range(self.M):
                competition_term_value += X_values[s_idx, 0] * (sum_all_X_scalar - X_values[s_idx, 0])
        
        competition_term_value *= self.beta

        energy = first_term + competition_term_value
        return energy