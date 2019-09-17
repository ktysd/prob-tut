import numpy as np
from scipy.integrate import ode

class class_EnKF: #アンサンブルカルマンフィルタ(Ensemble Kalman filter)

    def __init__(self, xdim, ydim, Q, R, pn):

        ### システムのサイズ
        self.xdim = xdim #状態の次元
        self.ydim = ydim #観測の次元
        self.pn   = pn   #アンサンブルの粒子数

        ### 雑音
        self.Q    = np.array(Q)         #システム雑音の共分散
        self.wdim = self.Q.shape[0]     #システム雑音の次元
        self.w    = np.zeros(self.wdim) #システム雑音ベクトル
        self.wav  = np.zeros(self.wdim) #システム雑音の平均 = 0

        self.R    = np.array(R)         #観測雑音の共分散
        self.vdim = self.R.shape[0]     #観測雑音の次元
        self.v    = np.zeros(self.vdim) #観測雑音ベクトル
        self.vav  = np.zeros(self.vdim) #観測雑音の平均 = 0
        
        ### アンサンブル行列
        self.Xp = np.zeros((self.xdim, self.pn))  #予測推定アンサンブル
        self.Xf = np.zeros_like(self.Xp)          #濾波アンサンブル
        self.Yp = np.zeros((self.ydim, self.pn))  #予測出力アンサンブル
        self.ones = np.ones((self.pn,1))
        
        self.bias = self.pn - 1
        self.yones = np.ones((self.ydim, self.pn))
    
    ### 小クラス用の初期化関数
    def system_definition(s, F_func, H_func, x0, P0, t0=0):

        s.F_func = F_func
        s.H_func = H_func
        
        s.init_ensemble(x0, P0) #アンサンブルの初期化
        s.xf = np.mean(s.Xp, axis=1) #濾波推定値の暫定初期値

    def state_eqn(s, x): #状態方程式の右辺
        s.update_w()
        return s.F_func(x, s.t) #s.t は filtering(yt,t) で更新された値
        
    def output_eqn(s, x): #観測方程式の右辺
        s.update_v()
        return s.H_func(x) + s.v
        
    def update_w(s): #システム雑音の更新
        if s.wdim == 1:
            s.w = np.sqrt(s.Q[0]) * np.random.randn() #正規乱数
        else:
            s.w = np.random.multivariate_normal(s.wav, s.Q)

    def update_v(s): #観測雑音
        if s.vdim == 1:
            s.v = np.sqrt(s.R[0])*(np.random.randn()) #正規乱数
        else:
            s.v = np.random.multivariate_normal(s.vav, s.R)
        
    ### アンサンブルの初期化と更新
    def init_ensemble(s, x0, P0):
        s.Xp = np.random.multivariate_normal(x0, P0, s.pn).T #(xdim)x(pn)ガウス行列  
        
    def update_Yp(s): #予測出力アンサンブルの更新
        s.Yp = np.apply_along_axis(s.output_eqn, 0, s.Xp)
        
    def update_Xp(s): #予測推定アンサンブルの更新
        s.Xp = np.apply_along_axis(s.state_eqn, 0, s.Xp)
        
    ### 濾波推定
    def filtering(s, yt, t, skip_prediction = False):
        
        ### フィルタ内の時刻の更新
        s.t = t
        
        ### 予測出力アンサンブルの計算
        s.update_Yp()

        ### カルマンゲインの計算
        s.meanXp = np.mean(s.Xp, axis=1).reshape(-1,1)
        s.meanYp = np.mean(s.Yp, axis=1).reshape(-1,1)
        s.covXY = s.bias*( np.dot(s.Xp, s.Yp.T)/s.pn - np.dot(s.meanXp, s.meanYp.T) )
        s.covYY = s.bias*( np.dot(s.Yp, s.Yp.T)/s.pn - np.dot(s.meanYp, s.meanYp.T) )
        s.K = np.dot(s.covXY, np.linalg.pinv(s.covYY))

        ### 濾波アンサンブルと濾波推定値の計算
        yt_matrix = np.dot(np.diag(yt), s.yones)
        s.Xf = s.Xp + np.dot(s.K, (yt_matrix - s.Yp)) #濾波アンサンブル
        s.xf = np.mean(s.Xf, axis=1) #濾波推定値
    
        if skip_prediction is not True:
            s.prediction() #予測推定

    ### 予測推定
    def prediction(s):
        s.Xp = s.Xf #濾波推定値を予測の初期値に
        s.update_Xp()
