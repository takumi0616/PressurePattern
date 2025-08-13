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

適正数値帯 : 87~88
クラスタ数 : 45~30

- 数値が低いとクラスタ数が増える
- 数値が高いとクラスタ数が減る
