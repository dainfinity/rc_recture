# Reservoir Computing (RC) レクチャー

## 概要

このリポジトリは、Echo State Network (ESN) を中心としたリザーバコンピューティングの実装とデモンストレーションを提供します。リザーバコンピューティングは、リカレントニューラルネットワークの訓練を効率化する機械学習手法で、時系列データの処理や予測タスクに特に有効です。

## 主な機能

### 1. Echo State Network (ESN) 実装
- PyTorchベースの柔軟なESN実装
- カスタマイズ可能なリザーバパラメータ
- 回帰・分類の両方に対応

### 2. ベンチマークタスク
- **NARMA タスク**: 非線形自己回帰移動平均タスクによる予測性能評価
- **STM タスク**: 短期記憶容量の定量的評価
- 各タスクに対する包括的な評価指標と可視化

### 3. 評価指標
- RMSE (Root Mean Square Error)
- NMSE (Normalized Mean Square Error)  
- R² (決定係数)
- MSE (Mean Square Error)
- Memory Capacity (記憶容量)

## ファイル構成

```
rc_recture/
├── README.md                 # このファイル
├── pyproject.toml           # プロジェクト設定・依存関係
├── uv.lock                  # 依存関係のロックファイル
├── requirements.txt         # pip用依存関係
├── hello.py                 # 簡単な動作確認用
└── esn/                     # ESN実装メインディレクトリ
    ├── model.py             # ESNモデル実装
    ├── narma_task.py        # NARMAタスク実装
    ├── stm_task.py          # STMタスク実装
    ├── examples.py          # 使用例・デモンストレーション
    ├── test_narma.py        # NARMAタスクテスト
    └── README.md            # ESN使用方法
```

## 環境構築

このプロジェクトは[uv](https://docs.astral.sh/uv/)を使用した依存関係管理を推奨しています。

### 1. uvのインストール

#### macOS・Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### pipx経由（全プラットフォーム対応）:
```bash
pipx install uv
```

### 2. プロジェクトのセットアップ

```bash
# リポジトリをクローン
git clone <repository-url>
cd rc_recture

# 依存関係の同期とvenv作成
uv sync

# 仮想環境をアクティベート
source .venv/bin/activate  # macOS/Linux
# または
.venv\Scripts\activate     # Windows
```

### 3. 動作確認

```bash
# 簡単な動作確認
python hello.py

# NARMAタスクの実行
python esn/narma_task.py

# STMタスクの実行
python esn/stm_task.py
```

## クイックスタート

### NARMA-10タスクの実行例

```python
from esn.narma_task import run_narma_experiment

# NARMA-10実験の実行
results = run_narma_experiment(
    order=10,                 # NARMA次数
    reservoir_size=100,       # リザーバサイズ  
    spectral_radius=0.95,     # スペクトル半径
    sequence_length=2000,     # データ長
    random_state=42,          # 再現性のための乱数シード
    plot_results=True         # 結果の可視化
)

print(f"RMSE: {results['rmse']:.6f}")
print(f"R²: {results['r2']:.6f}")
```

### 記憶容量評価の実行例

```python
from esn.stm_task import evaluate_memory_capacity

# 記憶容量の評価
results = evaluate_memory_capacity(
    reservoir_size=100,
    spectral_radius=0.95,
    sequence_length=2000,
    max_delay=50,
    random_state=42
)

print(f"Total Memory Capacity: {results['total_capacity']:.2f}")
```

## 主要パラメータの調整指針

### ESNパラメータ
- **reservoir_size**: 50-200（タスクの複雑さに応じて調整）
- **spectral_radius**: 0.9-0.99（記憶能力に大きく影響）
- **input_scaling**: 0.1-1.0（入力信号の強度調整）
- **connectivity**: 0.1-0.3（リザーバの結合密度）
- **leaking_rate**: 0.1-1.0（時間定数の調整）

### タスク固有パラメータ
- **washout**: 50-200（初期過渡状態の除去）
- **train_ratio**: 0.6-0.8（訓練・テスト分割比）
- **ridge_alpha**: 1e-8 ~ 1e-3（正則化強度）

## 期待される性能

### NARMA タスク
- **NARMA-5**: RMSE ~0.02-0.05, R² ~0.85-0.95
- **NARMA-10**: RMSE ~0.05-0.10, R² ~0.60-0.80

### STM タスク  
- **記憶容量**: リザーバサイズとおおむね線形関係
- **典型的な値**: 100ニューロンで30-50程度

## トラブルシューティング

### よくある問題

1. **NaN値の発生**
   - スペクトル半径を下げる（0.95以下）
   - 正則化パラメータを増やす
   - ウォッシュアウト期間を長くする

2. **性能が低い**
   - リザーバサイズを増やす
   - スペクトル半径を最適化
   - 入力スケーリングを調整

3. **計算が遅い**
   - リザーバサイズを減らす
   - シーケンス長を短くする
   - デバイスをGPUに変更（CUDA利用可能時）

## 依存関係

主要な依存パッケージ：
- torch >= 1.9.0
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- typing-extensions

詳細は`pyproject.toml`を参照してください。

## ライセンス

[ライセンス情報を記載]

## 参考文献

1. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
2. Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training.
3. Verstraeten, D., et al. (2007). An experimental unification of reservoir computing methods.

## 貢献

プルリクエストやイシューの報告を歓迎します。大きな変更を行う前に、まずイシューを作成して議論することをお勧めします。
