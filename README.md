# 気圧配置分類問題

# コマンド

## プログラム実行

```bash
nohup python main_v4.py > main_v4.log 2>&1 &
```

## タスクの削除

```bash
pkill -f "main_v4.py"
```

# 結果を mac に転送

## wsl-ubuntu → mac

```bash
rsync -avz --progress wsl-ubuntu:/home/takumi/docker_miniconda/src/PressurePattern/results_v4 /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/result_wsl-ubuntu
```

## via-tml2 → mac

```bash
rsync -avz --progress via-tml2:/home/s233319/docker_miniconda/src/PressurePattern/results_v4 /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/result_via-tml2
```

## gpu01 → mac

```bash
rsync -avz --progress gpu01:/home/devel/work_takasuka_git/docker_miniconda/src/PressurePattern/results_v4 /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/result_gpu01
```

## gpu02 → mac

```bash
rsync -avz --progress gpu02:/home/devel/work_takasuka_git/docker_miniconda/src/PressurePattern/results_v4 /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/result_gpu02
```
