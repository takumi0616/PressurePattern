# 気圧配置分類問題

# コマンド

## プログラム実行

```bash
nohup python main_v2.py > main_v2.log 2>&1 &
```

## タスクの削除

```bash
pkill -f "main_v2.py"
```

# パラメータ

83:16
82:18
81:25
80:43
79:53
77:

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
rsync -avz --progress gpu01:/home/devel/work_takasuka_git/docker_miniconda/src/PressurePattern/results_v2 /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern
```

rsync -avz --progress gpu01:/home/devel/work_takasuka_git/docker_miniconda/src/PressurePattern/3type_som/outputs_som_fullperiod /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/3type_som


rsync -avz --progress gpu01:/home/devel/work_takasuka_git/docker_miniconda/src/PressurePattern/results_v1 /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern