# 気圧配置分類問題

# コマンド

## プログラム実行

```bash
nohup python main_v3.py > main_v3.log 2>&1 &
```

## タスクの削除

```bash
pkill -f "main_v3.py"
```

# 結果を mac に転送

## wsl-ubuntu → mac

```bash
rsync -avz --progress wsl-ubuntu:/home/takumi/docker_miniconda/src/PressurePattern/results_v3 /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern
```

## via-tml2 → mac

```bash
rsync -avz --progress via-tml2:/home/s233319/docker_miniconda/src/PressurePattern/results_v3 /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern
```

## gpu01 → mac

```bash
rsync -avz --progress gpu01:/home/devel/work_takasuka_git/docker_miniconda/src/PressurePattern/results_v3 /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern
```

## gpu02 → mac

```bash
rsync -avz --progress gpu02:/home/devel/work_takasuka_git/docker_miniconda/src/PressurePattern/results_v3 /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern
```
