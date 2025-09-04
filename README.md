# 気圧配置分類（SOM, Multi-distance）

本プログラムは ERA5 の海面更正気圧（msl）から偏差を作り、SOM（自己組織化マップ）でクラスタリングします。  
距離は以下に対応しています（コード内 methods 参照）:

- euclidean（ユークリッド）
- ssim5（5x5 移動窓, C=0）
- s1（Teweles–Wobus S1）
- s1ssim / s1ssim5_hf / s1ssim5_and / pf_s1ssim（S1 と SSIM の合成）
- s1gssim / gssim（勾配構造類似）

学習後は代表ラベルや True Medoid、混同行列、各種メトリクスを CSV/PNG/ログに出力します。

データファイル:

- `./prmsl_era5_all_data_seasonal_large.nc`（同ディレクトリに配置）

# 実行方法（GPU/CPU 切替と複数同時実行）

main_v5.py に以下の引数を追加しました:

- `--seed INT` 乱数シード（デフォルト: コード側の SEED）
- `--gpu INT` 使う GPU 番号（0 や 1 など）。`--device` 指定がある場合は無視されます
- `--device STR` `'cpu'`, `'cuda'`, `'cuda:N'` を直接指定（例: `cuda:0`）
- `--result-dir PATH` 出力先ルートを明示指定（未指定なら `results_v5_iter{NUM}_batch{BATCH}_seed{SEED}_{devtag}` が自動生成）

優先順位:

1. `--device` を指定したらそれを使用
2. `--device` 未指定かつ `--gpu` 指定があれば `cuda:{gpu}` を使用
3. どちらも未指定なら、CUDA 利用可能時は `cuda:0`、なければ `cpu`

出力先:

- ルート: `RESULT_DIR`（引数で明示指定可、未指定時は seed と device から自動命名）
- 配下: `learning_result/` と `verification_results/`
- 実行全体ログ: `RESULT_DIR/run_v4.log`（既存命名を踏襲）
- 距離別ログ/CSV/PNG が個別フォルダに出ます

## 1GPU 環境（自動で GPU0 を使用）

```bash
# 前提: カレントは src/PressurePattern
python main_v5.py --seed 1
# あるいは明示
python main_v5.py --gpu 0 --seed 1
```

CPU 実行に切り替えたい場合:

```bash
python main_v5.py --device cpu --seed 1
```

出力例（自動命名）:

```
./results_v5_iter1000_batch256_seed1_cuda0/
  ├─ run_v4.log
  ├─ learning_result/
  └─ verification_results/
```

## 2GPU 環境（GPU の 0 番 と 1 番 を同時実行）

以下のように別のシード・別の GPU 番号を与えると、同時に 2 本走らせられます。  
1GPU 環境でも `--gpu 0` のみを使えば従来通り動作します。

- フォアグラウンドで開始（例）

```bash
python main_v5.py --gpu 0 --seed 1
python main_v5.py --gpu 1 --seed 2
```

- バックグラウンドで開始（nohup + ログ標準出力 redirection）

```bash
nohup python main_v5.py --gpu 0 --seed 1 > seed1_gpu0.out 2>&1 &
nohup python main_v5.py --gpu 1 --seed 2 > seed2_gpu1.out 2>&1 &
```

- notify-run を使う（通知が欲しい場合）

```bash
notify-run -- nohup python main_v5.py --gpu 0 --seed 1 > seed1_gpu0.out 2>&1 &
notify-run -- nohup python main_v5.py --gpu 1 --seed 2 > seed2_gpu1.out 2>&1 &
```

出力先を明示したい場合（同名衝突を避けたいとき等）:

```bash
nohup python main_v5.py --gpu 0 --seed 1 --result-dir ./results_gpu0_seed1 > s1_g0.out 2>&1 &
nohup python main_v5.py --gpu 1 --seed 2 --result-dir ./results_gpu1_seed2 > s2_g1.out 2>&1 &
```

補足:

- `--device cuda:0` / `--device cuda:1` と書いても同様に動作します（`--device` が優先されます）
- `CUDA が利用不可` の環境で `--device cuda:*` を渡した場合は自動で `cpu` にフォールバックします（警告出力あり）

## 代表的なプロセス管理

- 実行中のプロセス確認:

```bash
ps aux | grep main_v5.py | grep -v grep
```

- 全て停止（強制）:

```bash
pkill -f "main_v5.py"
```

- 指定 GPU だけを止めたい場合はコマンドラインを絞り込む:

```bash
pkill -f "main_v5.py.*--gpu 0"
# または明示 device 指定で動かしているなら
pkill -f "main_v5.py.*--device cuda:0"
```

- 個別に PID を kill（安全策）:

```bash
# 確認
ps aux | grep "main_v5.py" | grep -v grep
# 停止
kill -9 <PID>
```

# 出力物の場所と内容

- ルート: `RESULT_DIR`（例: `results_v5_iter1000_batch256_seed1_cuda0`）
  - `run_v4.log`: 実行全体ログ（開始/デバイス/メトリクス要約など）
  - `learning_result/*`: 各距離法ごとのログ・CSV・図
    - `*_results.log`（学習ログ、QE・MacroRecall・NodewiseMatchRate など）
    - `*_iteration_metrics.csv / .png`（学習イテレーションごとの履歴）
    - `*_assign_all.csv`（全サンプルの BMU 割当）
    - `*_som_node_avg_all.png`（ノード平均マップ）
    - `*_som_node_true_medoid_all.png` / `*_node_true_medoids.csv` など（True Medoid 関連）
    - ラベル分布図（base のみ）など
  - `verification_results/*`:
    - 検証混同行列, per-label 再現率 CSV, バー図
    - 検証データのノード平均マップ, ラベル分布図（base のみ）など

# よく使う実行例

- 1GPU 環境で seed を変えて順番に実行:

```bash
python main_v5.py --gpu 0 --seed 1
python main_v5.py --gpu 0 --seed 2
```

- 2GPU 環境で seed を変えて同時に 2 本:

```bash
nohup python main_v5.py --gpu 0 --seed 1 > s1_g0.out 2>&1 &
nohup python main_v5.py --gpu 1 --seed 2 > s2_g1.out 2>&1 &
```

- CPU で試す:

```bash
python main_v5.py --device cpu --seed 1
```

- 出力を明示し（分かりやすいフォルダ名で）保存:

```bash
python main_v5.py --gpu 0 --seed 7 --result-dir ./results_gpu0_seed7
```

# 結果を mac に転送（rsync 例）

プロジェクトルート例:

- gpu01/gpu02: `/home/devel/work_takasuka_git/docker_miniconda/src/PressurePattern`
- via-tml2: `/home/s233319/docker_miniconda/src/PressurePattern`
- wsl-ubuntu: `/home/takumi/docker_miniconda/src/PressurePattern`

フォルダ名は seed / device により変化します（例: `results_v5_iter1000_batch256_seed1_cuda0`）。  
パターンマッチや個別フォルダを指定して rsync してください。

- wsl-ubuntu → mac

```bash
rsync -avz --progress wsl-ubuntu:/home/takumi/docker_miniconda/src/PressurePattern/results_v5_* \
  /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/result_wsl-ubuntu
```

- via-tml2 → mac

```bash
rsync -avz --progress via-tml2:/home/s233319/docker_miniconda/src/PressurePattern/results_v5_* \
  /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/result_via-tml2
```

- gpu01 → mac

```bash
rsync -avz --progress gpu01:/home/devel/work_takasuka_git/docker_miniconda/src/PressurePattern/results_v5_* \
  /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/result_gpu01
```

- gpu02 → mac

```bash
rsync -avz --progress gpu02:/home/devel/work_takasuka_git/docker_miniconda/src/PressurePattern/results_v5_* \
  /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/result_gpu02
```

注意:

- `results_v5_*` のパターンで一括転送できます。個別に指定する場合は実行時に表示された `RESULT_DIR` をそのまま使うと確実です。
- リモート側のシェルのグロブ展開に依存するため、必要に応じて引用符/エスケープを調整してください。

# 変更点（今回の改善）

- GPU/CPU デバイス指定の CLI 対応
  - `--device`（例: `cuda:0`, `cuda:1`, `cpu`）
  - `--gpu`（整数で GPU 番号。`--device` 指定時は無視）
- 1GPU/2GPU 双方での同一コード動作
  - 2GPU 環境では `--gpu 0` と `--gpu 1` で同時に別条件実行が可能
  - 1GPU 環境では自動で `cuda:0`（ない場合は `cpu` にフォールバック）
- `--seed` と `--result-dir` を追加
  - シードや出力先を変えて複数ジョブを並走/整理しやすく
- True Medoid 計算等、内部で用いる torch デバイスをユーザ指定デバイスに統一
  - GPU 0/1 の使い分けが混在しないよう安全に反映
