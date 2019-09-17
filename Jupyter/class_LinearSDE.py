import numpy as np
from scipy.integrate import ode
from class_SDE import * #実行フォルダにSDE.pyを置いてインポートします．

class class_LinearSDE(class_SDE): #SDEの子クラス

    def __init__(self, A, D, C, Q, R, x0, dt, t0=0.0, B=None):

        ### システム行列
        self.A = np.array(A) #状態行列
        self.D = np.array(D) #駆動行列
        self.C = np.array(C) #観測行列
        # システムのサイズ
        xdim = A.shape[1] #状態の次元＝Aの列数
        ydim = C.shape[0] #観測の次元＝Cの行数
        wdim = D.shape[1] #システム雑音の次元＝Dの列数

        super().__init__(xdim, ydim, Q, R)

        ### 制御入力
        if B is not None:
            self.B = np.array(B)
            self.udim = self.B.shape[1] #制御入力の次元
            self.u = np.zeros(self.udim)
        else:
            self.udim = 0
                     
        ### 初期設定
        self.setup(x0, dt, t0)

    ### 確率微分方程式(SDE: stochastic differential equation)
    def StateEqn(s, t, x):
        Dww = np.ravel(s.D.dot(s.w)) #odeのベクトルは1次元配列

        dx0 = s.A.dot(x) + Dww
        
        if s.udim==0:
            dx = dx0
        else:
            Bu = np.ravel(s.B.dot(s.u)) #odeのベクトルは1次元配列
            dx = dx0 + Bu

        return dx

    def OutputEqn(s, x):
        return s.C.dot(x)
