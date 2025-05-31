# Echo State Network (ESN) Implementation

PyTorchベースの拡張可能なEcho State Network実装です。Short Term Memoryタスクによるメモリ容量評価機能を含みます。

## 特徴

- **拡張可能な設計**: モジュール化されたアーキテクチャで、新しいタスクや評価方法を簡単に追加可能
- **完全なPyTorch実装**: 自動微分とGPU加速をサポート
- **包括的な評価ツール**: Memory Capacity測定、パラメータ比較、ベンチマークツール
- **豊富な使用例**: 時系列予測、Mackey-Glass系列、STMタスクなどの実用例

## パッケージ構成

```
esn/
├── __init__.py          # パッケージ初期化
├── model.py             # 核心ESN実装
├── stm_task.py          # Short Term Memoryタスク
├── examples.py          # 使用例とベンチマーク
└── README.md           # このファイル
```

## インストール

必要な依存関係をインストール：

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

### Memory Capacity評価

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

### 遅延の詳細可視化

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

### パラメータ比較

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

## ESNパラメータ

| パラメータ | 説明 | 推奨範囲 |
|-----------|------|---------|
| `input_size` | 入力特徴数 | データに依存 |
| `reservoir_size` | リザーバーサイズ | 50-500 |
| `output_size` | 出力特徴数 | データに依存 |
| `spectral_radius` | スペクトル半径 | 0.9-0.99 |
| `input_scaling` | 入力スケーリング | 0.1-2.0 |
| `connectivity` | 接続密度 | 0.1-0.3 |
| `leaking_rate` | リーク率 | 0.1-1.0 |
| `bias_scaling` | バイアススケーリング | 0.0-1.0 |
| `noise_level` | ノイズレベル | 0.0-0.1 |

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

## Memory Capacity理論

Memory Capacityは、ESNが過去の入力をどの程度記憶できるかを定量化する指標です：

- **理論的最大値**: リザーバーサイズN
- **実際の値**: 通常N以下（パラメータ設定に依存）
- **計算方法**: 各遅延kについて、u(t-k)の復元精度（R²）を測定
- **総容量**: MC = Σ MC(k)

最適なESNでは、容量比（MC/N）が0.5-0.9程度になります。

## パフォーマンス最適化

### GPU使用

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

esn = EchoStateNetwork(
    input_size=1,
    reservoir_size=100,
    output_size=1,
    device=device
)
```

### バッチ処理

```python
# 複数系列の同時処理
batch_size = 32
x_batch = torch.randn(batch_size, sequence_length, input_size)
y_batch = torch.randn(batch_size, sequence_length, output_size)

esn.fit(x_batch, y_batch)
predictions = esn.predict(x_batch)
```

### 可視化の最適化

```python
# 見やすい可視化のための推奨設定
evaluate_memory_capacity(
    reservoir_size=100,
    sequence_length=800,       # 短めで高速
    max_delay=50,              # 適度な範囲
    train_ratio=0.7,           # 標準的な分割比
    plot_delays=[1, 5, 10],    # 主要な遅延のみ
)

# 時系列プロットの点数制限（デフォルト200点で見やすい）
plot_delay_accuracy(detailed_results, max_points=200)

# 散布図の点数制限（デフォルト1000点で適度な密度）
plot_scatter_accuracy(detailed_results, max_points=1000)
```

## 使用例の実行

パッケージに含まれる使用例を実行：

```python
# 基本的な使用例
python examples.py

# STMタスクのみ
python stm_task.py

# 個別の例
from esn.examples import example_simple_prediction, example_mackey_glass
example_simple_prediction()
example_mackey_glass()
```

## 拡張ガイド

### 新しいアクティベーション関数

```python
class CustomESN(EchoStateNetwork):
    def forward(self, input_data, reset_state=True):
        # 親クラスのforward()をベースに、
        # アクティベーション関数部分をカスタマイズ
        # new_state = your_activation(input_activation + reservoir_activation)
        pass
```

### 新しいタスク

```python
def your_custom_task():
    # データ生成
    x, y = generate_your_data()
    
    # ESN作成・学習
    esn = EchoStateNetwork(...)
    esn.fit(x, y)
    
    # 評価
    predictions = esn.predict(x_test)
    score = your_evaluation_metric(y_test, predictions)
    
    return score
```

## トラブルシューティング

### よくある問題

1. **不安定な学習**: スペクトル半径を下げる（< 1.0）
2. **低いメモリ容量**: 接続密度やリザーバーサイズを増やす
3. **過学習**: リッジ回帰の正則化パラメータを増やす
4. **遅い収束**: washout期間を延ばす

### パフォーマンス調整

- **スペクトル半径**: 0.95-0.99（エッジオブカオス）
- **接続密度**: 0.1-0.3（疎接続）
- **入力スケーリング**: 0.5-2.0（入力のダイナミクスに依存）
- **リーク率**: 1.0（メモリタスク）、< 1.0（高速変化タスク）

## 参考文献

1. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. GMD Report 148.
2. Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training. Computer Science Review, 3(3), 127-149.
3. Verstraeten, D., et al. (2007). An experimental unification of reservoir computing methods. Neural networks, 20(3), 391-403.

## ライセンス

MIT License 