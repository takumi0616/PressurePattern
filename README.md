# ACS (Adaptive Competitive Self-organizing) アルゴリズム

Python による適応的競争自己組織化（ACS）アルゴリズムの実装です。本実装は、リアルタイムクラスタリングとベクトル量子化のための新しい自己組織化モデルを提供します。

## 参考論文

本実装は以下の論文に基づいています：

> Zahra Sarafraz, Hossein Sarafraz and Mohammad R Sayeh (2018). "Real-time classifier based on adaptive competitive self-organizing algorithm", _Adaptive Behavior_, Vol. 26(1) 21–31. DOI: [10.1177/1059712318760695](https://doi.org/10.1177/1059712318760695)

### 論文の主な特徴

- **動的構造と自己調整パラメータ**: モデルは外部制御機構を必要としない教師なし分類器として機能します
- **寄生的限界点問題への対処**: より正確なラベル割り当てを実現します
- **エネルギー関数ベース**: Lyapunov 関数形式のエネルギー関数により、有限個の孤立平衡点での安定化を保証します
- **競争メカニズム**: Lotka-Volterra 競争排除に基づく競争メカニズムと勾配降下理論を組み合わせています

## ディレクトリ構成

```
ACS/
├── acs.py                    # ACSアルゴリズムの主要実装
├── test_acs_circular.py      # 円形活性化関数を使用したテスト
├── test_acs_elliptical.py    # 楕円形活性化関数を使用したテスト
└── README.md                 # このファイル
```

## 必要なライブラリ

```python
numpy
scikit-learn
matplotlib
seaborn
scipy
japanize_matplotlib  # 日本語表示用（オプション）
pandas  # CSV出力用（オプション）
```

インストール方法：

```bash
pip install numpy scikit-learn matplotlib seaborn scipy japanize-matplotlib pandas
```

## acs.py の使い方

### 基本的な使用方法

```python
from acs import ACS
import numpy as np

# ACSモデルのインスタンス化
acs_model = ACS(
    gamma=1.0,                     # エネルギー関数の目標値パラメータ
    beta=0.1,                      # 競争係数
    learning_rate_W=0.01,          # ラベルWの学習率
    learning_rate_lambda=0.001,    # 警戒パラメータλの学習率
    learning_rate_Z=0.01,          # 深さパラメータZの学習率
    max_clusters=10,               # 最大クラスタ数
    initial_clusters=3,            # 初期クラスタ数
    activation_type='circular',    # 活性化関数タイプ ('circular' or 'elliptical')
    random_state=42               # 再現性のためのシード値
)

# データの学習
X_train = np.random.rand(100, 4)  # 100サンプル、4次元の例
acs_model.fit(X_train, epochs=50)

# クラスタの予測
predictions = acs_model.predict(X_train)

# クラスタ中心の取得
cluster_centers = acs_model.get_cluster_centers()
```

### 主要なパラメータ

- **gamma (γ)**: エネルギー関数の目標値に関連するパラメータ。論文式(9)参照
- **beta (β)**: 競争係数。Winner-Takes-All 機構の強さを制御
- **learning_rate_W, learning_rate_lambda, learning_rate_Z**: 各パラメータの学習率
- **activation_type**:
  - `'circular'`: 円形の活性化関数（論文式(2)）
  - `'elliptical'`: 楕円形の活性化関数（論文式(6)）
- **theta_new**: 新規クラスタ生成の閾値
- **death_patience_steps**: クラスタ削除までの非活性許容ステップ数
- **Z_death_threshold**: クラスタ削除の Z 値閾値

### 動的クラスタリング機能

本実装では論文の概念に基づき、以下の動的機能を実現しています：

1. **クラスタの動的生成**: 新しい入力パターンが既存クラスタと十分に類似していない場合、新しいクラスタを生成
2. **クラスタの動的削除**: 寄生的アトラクタ（非活性クラスタ）の自動削除

## test_acs_circular.py / test_acs_elliptical.py の使い方

これらのテストスクリプトは、Iris データセットを使用して ACS アルゴリズムの性能を評価します。

### 実行方法

```bash
# 円形活性化関数でのテスト
python test_acs_circular.py

# 楕円形活性化関数でのテスト
python test_acs_elliptical.py
```

### テストスクリプトの機能

1. **ランダムサーチによるハイパーパラメータ最適化**: 最大 5000 回の試行で最適なパラメータ組み合わせを探索
2. **性能評価指標**:
   - Adjusted Rand Index (ARI)
   - マッピング後の Accuracy
   - 混同行列
3. **可視化**:
   - PCA 空間でのクラスタリング結果
   - エポック毎の評価指標推移
   - エネルギー関数の等高線図
   - 混同行列のヒートマップ

### 出力ファイル

各テストスクリプトは以下のファイルを生成します：

```
result_test_acs_[circular/elliptical]_random_YYYYMMDD_HHMMSS/
├── random_search_log_[circular/elliptical]_YYYYMMDD_HHMMSS.txt     # 実行ログ
├── random_search_all_results_[circular/elliptical]_YYYYMMDD_HHMMSS.csv  # 全試行結果
├── best_model_params_dynamic_[circular/elliptical]_YYYYMMDD_HHMMSS.txt  # 最良パラメータ
├── metrics_vs_epoch_best_dynamic_[circular/elliptical]_YYYYMMDD_HHMMSS.png  # 評価指標推移
├── confusion_matrix_best_dynamic_[circular/elliptical]_YYYYMMDD_HHMMSS.png  # 混同行列
├── pca_clustering_best_dynamic_[circular/elliptical]_YYYYMMDD_HHMMSS.png    # PCAクラスタリング
└── energy_contour_plot_[circular/elliptical]_YYYYMMDD_HHMMSS.png          # エネルギー等高線図
```

## 実装の特徴

### 論文の主要概念の実装

1. **エネルギー関数**: 式(9)に基づくエネルギー関数の実装

   ```
   E = [γ - Σ(Z_j × X_j)]² + β × Σ(X_s × X_j) (s≠j)
   ```

2. **活性化関数**:

   - 円形: X_jp = 1 / (1 + λ_j × ||U_p - W_j||²)
   - 楕円形: より複雑な形状の谷を形成可能

3. **動的パラメータ更新**:
   - W_ij（ラベル）: 勾配降下法による更新
   - λ（警戒パラメータ）: クラスタサイズの動的調整
   - Z_j（深さパラメータ）: 勝者は 1 へ、敗者は 0 へ収束

### 性能

論文 Table 2 の結果と比較：

- Iris データセット（円形）: Accuracy 93%
- Iris データセット（楕円形）: Accuracy 95.3%

## 注意事項

- 初回実行時は、パラメータのランダムサーチに時間がかかる場合があります
- `max_random_trials`パラメータを調整することで、探索時間を制御できます
- 結果の再現性のため、`random_state`パラメータの設定を推奨します

## ライセンス

本実装は研究・教育目的での使用を想定しています。商用利用の際は、参考論文の著者への確認を推奨します。

## 参考

```
@article{sarafraz2018real,
  title={Real-time classifier based on adaptive competitive self-organizing algorithm},
  author={Sarafraz, Zahra and Sarafraz, Hossein and Sayeh, Mohammad R},
  journal={Adaptive Behavior},
  volume={26},
  number={1},
  pages={21--31},
  year={2018},
  publisher={SAGE Publications Sage UK: London, England}
}
```

```bash
nohup python multi_prmsl_acs_random_v3.py > output.log 2>&1 &
```
