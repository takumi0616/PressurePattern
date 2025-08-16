# コマンド

## プログラム実行

```bash
nohup python main_v2.py > main_v2.log 2>&1 &

nohup python main_v3.py > main_v3.log 2>&1 &

nohup python main_v4.py > main_v4.log 2>&1 &
```

## タスクの削除

```bash
pkill -f "main_v2.py"

pkill -f "main_v3.py"

pkill -f "main_v4.py"
```

# プログラム説明

## v2(SSIM スコアによる HAC+k-medoids) の TH_MERGE

適正数値帯 : 0.4~0.5
クラスタ数 : 20~55

- 数値が低いとクラスタ数が減る
- 数値が高いとクラスタ数が増える

## v3(S1 スコアによる HAC+k-medoids) の TH_MERGE

適正数値帯 : 80~85
クラスタ数 : 1206~11

- 数値が低いとクラスタ数が増える
- 数値が高いとクラスタ数が減る

# 結果を mac に転送

## wsl-ubuntu → mac

```bash
rsync -avz --progress wsl-ubuntu:/home/takumi/docker_miniconda/src/PressurePattern/HAC+k-medoids/v4_clustering_results_83 /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/HAC+k-medoids/document
```

## via-tml2 → mac

```bash
rsync -avz --progress via-tml2:/home/s233319/docker_miniconda/src/PressurePattern/HAC+k-medoids/v4_clustering_results_84 /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/HAC+k-medoids/document
```

## gpu01 → mac

```bash
rsync -avz --progress gpu01:/home/devel/work_takasuka_git/docker_miniconda/src/PressurePattern/clustering+som/results_v1 /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/document
```
