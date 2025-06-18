import numpy as np

# NARMAモデル
class NARMA:
    # パラメータの設定
    def __init__(self, m, a1, a2, a3, a4):
        self.m = m
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

    def generate_data(self, T, y_init, seed=0):
        n = self.m
        y = y_init
        np.random.seed(seed=seed)
        u = np.random.uniform(0, 0.5, T)

        # 時系列生成
        while n < T:
            # スケーリングを適用
            y_n = self.a1 * y[n-1] + self.a2 * y[n-1] * (np.sum(y[n-self.m:n-1]) / self.m) \
                + self.a3 * u[n-self.m] * u[n] + self.a4
            y.append(y_n)
            n += 1

        return u, np.array(y)
    
# generate NARMA train/valid/test data
def generate_narma_data(order, T_train, T_valid, T_washout, seed=0):
    """
    NARMAデータ生成と訓練・検証データへの分割を行う関数
    washoutは訓練に含まれるので自分で捨てる.
    """
    
    T_washout_narma = 10*order
    T_total = T_train + T_valid + T_washout + T_washout_narma
    y_init = [0.0] * order
    narma = NARMA(order, 0.3, 0.05, 1.5, 0.1)
    u, y = narma.generate_data(T_total, y_init, seed=seed)
    
    # 訓練データと検証データに分割（ギャップを含む）
    train_U = u[T_washout_narma:T_train + T_washout + T_washout_narma].reshape(-1, 1)
    train_Y = y[T_washout_narma:T_train + T_washout + T_washout_narma].reshape(-1, 1)
    valid_U = u[T_train + T_washout + T_washout_narma:T_train + T_valid + T_washout + T_washout_narma].reshape(-1, 1)
    valid_Y = y[T_train + T_washout + T_washout_narma:T_train + T_valid + T_washout + T_washout_narma].reshape(-1, 1)
    
    return train_U, train_Y, valid_U, valid_Y
