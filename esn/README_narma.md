# NARMA Task Implementation for Echo State Networks

## 概要

このプロジェクトでは、Echo State Network (ESN) の評価のためのNARMA（Non-linear Auto-Regressive Moving Average）タスクを実装しています。NARMAタスクは、リザーバコンピューティングシステムの計算能力を評価するための標準的なベンチマークです。

## NARMAタスクとは

NARMA-m タスクは以下の非線形自己回帰移動平均方程式に基づいています：

```
y(n+1) = 0.3*y(n) + 0.05*y(n)*sum(y(n-i), i=0 to m-1) + 1.5*u(n-m)*u(n) + 0.1
```

ここで：
- `y(n)` は時刻nでの出力
- `u(n)` は時刻nでの入力
- `m` はNARMAシステムの次数

## ファイル構成

- `narma_task.py`: メインのNARMAタスク実装
- `test_narma.py`: テストスクリプト
- `model.py`: ESNモデルの実装（既存）

## 主な機能

### 1. データ生成
- `generate_narma_data()`: NARMA時系列データの生成
- 数値安定性のためのクリッピング機能付き

### 2. 評価指標
以下の評価指標を計算します：
- **RMSE** (Root Mean Square Error): 平方根平均二乗誤差
- **NMSE** (Normalized Mean Square Error): 正規化平均二乗誤差
- **R²** (Coefficient of determination): 決定係数
- **MSE** (Mean Square Error): 平均二乗誤差

### 3. 可視化
- 時系列予測結果の比較
- 真値 vs 予測値の散布図
- 時間に対する絶対誤差の変化
- 評価指標の表示

### 4. 比較実験
- 異なるNARMA次数での性能比較
- 複数パラメータでの実験

## 使用方法

### 基本的な使用例

```python
from narma_task import run_narma_experiment

# NARMA-10 タスクの実行
results = run_narma_experiment(
    order=10,                 # NARMA次数
    reservoir_size=100,       # リザーバサイズ
    spectral_radius=0.95,     # スペクトル半径
    sequence_length=2000,     # データ長
    train_ratio=0.7,          # 訓練データの割合
    washout=100,              # ウォッシュアウト期間
    random_state=42,          # 乱数シード
    plot_results=True         # 結果の可視化
)

print(f"RMSE: {results['rmse']:.6f}")
print(f"R²: {results['r2']:.6f}")
```

### 複数次数の比較

```python
from narma_task import compare_narma_orders

# 異なるNARMA次数での比較
comparison_results = compare_narma_orders(
    orders=[3, 5, 8, 10],
    reservoir_size=100,
    sequence_length=1500,
    random_state=42
)
```

### コマンドラインからの実行

```bash
# メインのデモンストレーション
python esn/narma_task.py

# テストスクリプトの実行
python esn/test_narma.py
```

## パラメータの説明

### ESNパラメータ
- `reservoir_size`: リザーバのニューロン数（推奨: 50-200）
- `spectral_radius`: リザーバ重み行列のスペクトル半径（推奨: 0.9-0.99）
- `input_scaling`: 入力スケーリング（推奨: 0.1-1.0）
- `connectivity`: リザーバの結合率（推奨: 0.1-0.3）
- `leaking_rate`: リーキング率（推奨: 0.1-1.0）

### タスクパラメータ
- `order`: NARMA次数（高次ほど困難。推奨: 3-20）
- `sequence_length`: 時系列の長さ（推奨: 1000-5000）
- `train_ratio`: 訓練データの割合（推奨: 0.6-0.8）
- `washout`: 初期の無視する時間ステップ（推奨: 50-200）

## 期待される結果

### NARMA-5
- RMSE: ~0.02-0.05
- R²: ~0.85-0.95

### NARMA-10
- RMSE: ~0.05-0.10
- R²: ~0.60-0.80

### NARMA-15以上
- より困難で、性能は次数とともに低下

## トラブルシューティング

### NaN値が発生する場合
- NARMA次数を下げる（15以下を推奨）
- リザーバサイズを増やす
- 正則化パラメータ `ridge_alpha` を調整

### 性能が悪い場合
- スペクトル半径を調整（0.9-0.99）
- リザーバサイズを増やす
- 入力スケーリングを調整
- ウォッシュアウト期間を調整

## 実装の特徴

- **数値安定性**: 値のクリッピングによる爆発的成長の防止
- **エラーハンドリング**: NaN/inf値の検出と適切な処理
- **可視化**: 包括的な結果の可視化
- **再現性**: 乱数シードによる結果の再現可能性

## 参考文献

1. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
2. Atiya, A. F., & Parlos, A. G. (2000). New results on recurrent network training: unifying the algorithms and accelerating convergence.
3. Lukosevicius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training. 