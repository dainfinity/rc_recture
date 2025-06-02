# Echo State Network (ESN) 実装

PyTorchベースの拡張可能なEcho State Network実装です。Short Term Memory (STM) タスクとNARMAタスクによる包括的な評価機能を含んでいます。

## 特徴

- **拡張可能な設計**: モジュール化されたアーキテクチャにより、新しいタスクや評価方法を簡単に追加できます
- **完全なPyTorch実装**: 自動微分とGPU加速をサポートしています
- **包括的な評価ツール**: Memory Capacity測定、NARMAタスク、パラメータ比較、ベンチマークツールを提供します
- **豊富な使用例**: 時系列予測、Mackey-Glass系列、STMタスク、NARMAタスクなどの実用例が含まれています

## パッケージ構成

```
esn/
├── __init__.py          # パッケージ初期化
├── model.py             # 核心となるESN実装
├── stm_task.py          # Short Term Memoryタスク
├── narma_task.py        # NARMAタスク実装
├── test_narma.py        # NARMAタスクテスト
├── examples.py          # 使用例とベンチマーク
└── README.md           # このファイル
```

## インストール

必要な依存関係をインストールしてください：

```bash
pip install torch numpy matplotlib
```

## 基本的な使用方法

### ESNの作成と学習

```python
from esn import EchoStateNetwork

# ESNの作成
esn = EchoStateNetwork(
    input_size=1,
    reservoir_size=100,
    output_size=1,
    spectral_radius=0.95,
    input_scaling=1.0,
    connectivity=0.1
)

# データの準備（例：正弦波）
import torch
t = torch.linspace(0, 10, 1000).unsqueeze(0).unsqueeze(-1)
x = torch.sin(t)
y = torch.sin(t + 0.5)  # 遅延した正弦波

# 学習
esn.fit(x, y, washout=50)

# 予測
predictions = esn.predict(x)
```

## ベンチマークタスク

### 1. Short Term Memory (STM) タスク

#### Memory Capacity評価

```python
from esn import evaluate_memory_capacity

# メモリ容量評価
results = evaluate_memory_capacity(
    reservoir_size=100,
    spectral_radius=0.95,
    sequence_length=1000,      # 短めの系列長で高速評価
    max_delay=80,              # 適度な遅延数
    train_ratio=0.7,           # 訓練:検証 = 7:3
    plot_delays=[1, 5, 10]     # 特定の遅延を可視化
)

print(f"Total Memory Capacity: {results['total_capacity']:.2f}")
print(f"Capacity Ratio: {results['capacity_ratio']:.2f}")
```

#### 遅延の詳細可視化

```python
from esn import plot_delay_accuracy, plot_scatter_accuracy

# 時系列での真値と予測値の比較（見やすいように200ポイントに制限）
plot_delay_accuracy(
    results['detailed_results'], 
    delay_indices=[1, 5], 
    max_points=200
)

# 散布図での相関確認（1000ポイントまで表示）
plot_scatter_accuracy(
    results['detailed_results'], 
    delay_indices=[1, 5],
    max_points=1000
)
```

### 2. NARMA (Non-linear Auto-Regressive Moving Average) タスク

#### NARMAタスクについて

NARMA-m タスクは、以下の非線形自己回帰移動平均方程式に基づく標準的なベンチマークです：

```
y(n+1) = 0.3*y(n) + 0.05*y(n)*sum(y(n-i), i=0 to m-1) + 1.5*u(n-m)*u(n) + 0.1
```

ここで：
- `y(n)` は時刻nでの出力
- `u(n)` は時刻nでの入力
- `m` はNARMAシステムの次数

#### 基本的な使用例

```python
from esn.narma_task import run_narma_experiment

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
print(f"NMSE: {results['nmse']:.6f}")
print(f"R²: {results['r2']:.6f}")
```

#### 複数次数の比較

```python
from esn.narma_task import compare_narma_orders

# 異なるNARMA次数での比較
comparison_results = compare_narma_orders(
    orders=[3, 5, 8, 10],
    reservoir_size=100,
    sequence_length=1500,
    random_state=42
)
```

#### コマンドラインからの実行

```bash
# メインのデモンストレーション
python esn/narma_task.py

# テストスクリプトの実行
python esn/test_narma.py
```

## パラメータ設定

### ESNパラメータ

| パラメータ | 説明 | STM推奨範囲 | NARMA推奨範囲 |
|-----------|------|------------|---------------|
| `input_size` | 入力特徴数 | データに依存 | 1 |
| `reservoir_size` | リザーバサイズ | 50-500 | 50-200 |
| `output_size` | 出力特徴数 | データに依存 | 1 |
| `spectral_radius` | スペクトル半径 | 0.9-0.99 | 0.9-0.99 |
| `input_scaling` | 入力スケーリング | 0.1-2.0 | 0.1-1.0 |
| `connectivity` | 接続密度 | 0.1-0.3 | 0.1-0.3 |
| `leaking_rate` | リーク率 | 0.1-1.0 | 0.1-1.0 |
| `bias_scaling` | バイアススケーリング | 0.0-1.0 | 0.0-1.0 |
| `noise_level` | ノイズレベル | 0.0-0.1 | 0.0-0.1 |

### タスク固有パラメータ

#### STMタスク
- `sequence_length`: 時系列の長さ（推奨: 1000-5000）
- `max_delay`: 最大遅延（推奨: 50-200）
- `train_ratio`: 訓練データの割合（推奨: 0.6-0.8）
- `washout`: ウォッシュアウト期間（推奨: 50-200）

#### NARMAタスク
- `order`: NARMA次数（高次ほど困難。推奨: 3-20）
- `sequence_length`: 時系列の長さ（推奨: 1000-5000）
- `train_ratio`: 訓練データの割合（推奨: 0.6-0.8）
- `washout`: 初期の無視する時間ステップ（推奨: 50-200）

## 期待される性能

### STMタスク
- **記憶容量**: リザーバサイズとおおむね線形関係
- **典型的な値**: 100ニューロンで30-50程度
- **容量比**: 0.3-0.5（MC/リザーバサイズ）

### NARMAタスク
- **NARMA-5**: RMSE ~0.02-0.05, R² ~0.85-0.95
- **NARMA-10**: RMSE ~0.05-0.10, R² ~0.60-0.80
- **NARMA-15以上**: より困難で、性能は次数とともに低下

## 高度な使用例

### 回帰タスク用ESN

```python
from esn import ESNRegressor

esn_reg = ESNRegressor(
    input_size=1,
    reservoir_size=200,
    output_size=1,
    spectral_radius=0.95
)

# 学習と評価
esn_reg.fit(x_train, y_train)
mse = esn_reg.score(x_test, y_test, metric='mse')
r2 = esn_reg.score(x_test, y_test, metric='r2')
```

### 分類タスク用ESN

```python
from esn import ESNClassifier

esn_clf = ESNClassifier(
    input_size=features,
    reservoir_size=100,
    output_size=num_classes,
    spectral_radius=0.9
)

# 学習
esn_clf.fit(x_train, y_train_onehot)

# 予測
probabilities = esn_clf.predict_proba(x_test)
predictions = esn_clf.predict_classes(x_test)
```

### パラメータ比較実験

```python
from esn import compare_esn_parameters, plot_parameter_comparison

# 複数のスペクトル半径を比較
parameter_ranges = {
    'spectral_radius': [0.1, 0.5, 0.9, 0.95, 0.99, 1.1]
}

comparison_results = compare_esn_parameters(
    parameter_ranges=parameter_ranges,
    reservoir_size=100
)

# 結果をプロット
plot_parameter_comparison(comparison_results)
```

### マルチステップ予測

```python
# 初期入力
x_initial = x_test[:, :1, :]
predictions = []

# マルチステップ予測
for step in range(prediction_horizon):
    y_pred = esn.predict(x_initial, reset_state=False)
    predictions.append(y_pred[:, -1:, :])
    x_initial = y_pred[:, -1:, :]  # 予測を次の入力として使用

predictions = torch.cat(predictions, dim=1)
```

## 評価指標

### STMタスク
- **Memory Capacity (MC)**: 各遅延での復元精度（R²）の総和
- **Capacity Ratio**: MC / リザーバサイズ

### NARMAタスク
- **RMSE** (Root Mean Square Error): 平方根平均二乗誤差
- **NMSE** (Normalized Mean Square Error): 正規化平均二乗誤差
- **R²** (Coefficient of determination): 決定係数
- **MSE** (Mean Square Error): 平均二乗誤差

## Memory Capacity理論

Memory Capacityは、ESNが過去の入力をどの程度記憶できるかを定量化する指標です：

- **理論的最大値**: リザーバサイズN
- **実際の値**: 通常N以下（パラメータ設定に依存）
- **計算方法**: 各遅延kについて、u(t-k)の復元精度（R²）を測定
- **総容量**: MC = Σ MC(k)

最適なESNでは、容量比（MC/N）が0.3-0.5程度になります。

## トラブルシューティング

### 共通の問題

1. **NaN値の発生**
   - スペクトル半径を下げる（0.95以下）
   - 正則化パラメータ `ridge_alpha` を増やす
   - ウォッシュアウト期間を長くする

2. **性能が低い**
   - リザーバサイズを増やす
   - スペクトル半径を最適化（0.9-0.99）
   - 入力スケーリングを調整

3. **計算が遅い**
   - リザーバサイズを減らす
   - シーケンス長を短くする
   - デバイスをGPUに変更（CUDA利用可能時）

### NARMA固有の問題

1. **高次NARMAでNaN値**
   - NARMA次数を下げる（15以下を推奨）
   - リザーバサイズを増やす

2. **予測精度が悪い**
   - ウォッシュアウト期間を調整
   - 訓練データ比率を調整

## 実装の特徴

- **数値安定性**: 値のクリッピングによる爆発的成長の防止（NARMA）
- **エラーハンドリング**: NaN/inf値の検出と適切な処理
- **包括的な可視化**: 時系列比較、散布図、誤差変化、評価指標表示
- **再現性**: 乱数シードによる結果の再現可能性
- **拡張性**: 新しいタスクの追加が容易な設計

## パフォーマンス最適化

### GPU使用

```python
# GPU利用可能時
device = 'cuda' if torch.cuda.is_available() else 'cpu'
esn = EchoStateNetwork(..., device=device)
```

### バッチ処理

```python
# 複数系列の同時処理
batch_size = 10
inputs = torch.randn(batch_size, sequence_length, input_size)
esn.reset_state(batch_size)
```

### メモリ効率化

- 大きなデータセットでは `sequence_length` を分割
- 不要な中間状態は削除
- `washout` を適切に設定して無駄な計算を削減

## 参考文献

1. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
2. Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training.
3. Verstraeten, D., et al. (2007). An experimental unification of reservoir computing methods.
4. Atiya, A. F., & Parlos, A. G. (2000). New results on recurrent network training: unifying the algorithms and accelerating convergence.

## ライセンス

[ライセンス情報を記載] 