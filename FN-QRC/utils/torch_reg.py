import torch
import numpy as np

class TorchLinearRegression:
    """
    PyTorchによるGPU対応線形回帰クラス（擬似逆行列実装）
    """
    
    def __init__(self, alpha=0.01):
        
        self.alpha = alpha
        self.coef_ = None  # 重み係数
        self.device = None  # 計算デバイス
        
    def fit(self, X, y):
        """
        擬似逆行列を用いて線形回帰モデルを学習
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            訓練データの特徴量（すでに切片列を含む）
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            訓練データの目的変数
            
        Returns:
        --------
        self : returns an instance of self.
        """
        # デバイスの確認と保存
        if torch.is_tensor(X):
            self.device = X.device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tensorへの変換
        X_tensor = self._convert_to_tensor(X)
        y_tensor = self._convert_to_tensor(y)
        
        # yの次元数を確認し、必要に応じて調整
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(1)
        
        # 擬似逆行列の計算
        if self.alpha == 0:
            # SVD分解による方法（より安定）
            U, S, V = torch.linalg.svd(X_tensor, full_matrices=False)
            
            # 数値安定性のためのカットオフ
            eps = 1e-10 * torch.max(S)
            S_inv = torch.zeros_like(S)
            S_inv[S > eps] = 1.0 / S[S > eps]
            
            # 擬似逆行列の計算
            X_pinv = V.T @ torch.diag(S_inv) @ U.T
            
            # 回帰係数の計算
            self.coef_ = X_pinv @ y_tensor
        else:
            # X.T @ X + λI の計算（正則化項を追加）
            XTX = X_tensor.T @ X_tensor
            reg_term = self.alpha * torch.eye(XTX.size(0), device=XTX.device, dtype=XTX.dtype)
            XTX_reg = XTX + reg_term
            
            X_inv = torch.linalg.inv(XTX_reg)
            self.coef_ = X_inv @ X_tensor.T @ y_tensor
        
        return self
    
    def predict(self, X):
        """
        学習したモデルを使用して予測を行う
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            予測する特徴量（すでに切片列を含む）
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,) or (n_samples, n_targets)
            予測値
        """
        if self.coef_ is None:
            raise ValueError("モデルが学習されていません。先にfit()を呼び出してください。")
        
        # Tensorへの変換
        X_tensor = self._convert_to_tensor(X)
        
        # 予測の計算
        y_pred = X_tensor @ self.coef_
        
        # 出力が1次元の場合は次元を縮小
        if y_pred.size(1) == 1:
            y_pred = y_pred.squeeze(1)
        
        return y_pred
    
    def _convert_to_tensor(self, X):
        """データをTensor型に変換"""
        if not torch.is_tensor(X):
            X = torch.tensor(X, device=self.device, dtype=torch.float64)
        elif X.device != self.device:
            X = X.to(self.device)
        return X
    
    def get_params(self):
        """モデルのパラメータを返す"""
        return {
            'coef_': self.coef_.cpu().numpy() if self.coef_ is not None else None
        }


def r2_score(y_true, y_pred):
    """
    決定係数 R^2 を計算
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        真の値
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        予測値
        
    Returns:
    --------
    r2 : float
        決定係数 R^2 の値 (最大値は1.0)
    """
    # 同じデバイスに変換
    if torch.is_tensor(y_true) and torch.is_tensor(y_pred):
        if y_true.device != y_pred.device:
            y_pred = y_pred.to(y_true.device)
    elif torch.is_tensor(y_true) and not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred, device=y_true.device, dtype=y_true.dtype)
    elif not torch.is_tensor(y_true) and torch.is_tensor(y_pred):
        y_true = torch.tensor(y_true, device=y_pred.device, dtype=y_pred.dtype)
    else:
        # どちらもテンソルでない場合
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        y_true = torch.tensor(y_true, device=device, dtype=torch.float64)
        y_pred = torch.tensor(y_pred, device=device, dtype=torch.float64)
    
    # 次元の調整
    if y_true.dim() == 1 and y_pred.dim() == 2:
        y_true = y_true.unsqueeze(1)
    elif y_true.dim() == 2 and y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(1)
    
    # 各次元ごとのR2スコアを計算
    if y_true.dim() == 1:  # 一次元の場合
        y_mean = torch.mean(y_true)
        ss_tot = torch.sum((y_true - y_mean) ** 2)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        
        # ゼロ除算を防ぐ
        if ss_tot == 0:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
    else:  # 多次元の場合
        n_outputs = y_true.size(1)
        r2_sum = 0.0
        
        for i in range(n_outputs):
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]
            y_mean_i = torch.mean(y_true_i)
            
            ss_tot_i = torch.sum((y_true_i - y_mean_i) ** 2)
            ss_res_i = torch.sum((y_true_i - y_pred_i) ** 2)
            
            # ゼロ除算を防ぐ
            if ss_tot_i == 0:
                r2_i = 0.0
            else:
                r2_i = 1 - (ss_res_i / ss_tot_i)
            
            r2_sum += r2_i
        
        # 多次元の場合は平均をとる
        r2 = r2_sum / n_outputs
    
    # スカラー値として返す
    return r2.item()
