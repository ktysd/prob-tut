import numpy as np
from scipy.linalg import solve_discrete_are

class class_ssKF: # Steady-state Kalman filter
    
    def __init__(s, F, G, H, Q, R, x0, cov0=None):
        
        ### システム行列
        s.F = np.array(F) #状態推移行列
        s.G = np.array(G) #駆動行列
        s.H = np.array(H) #観測行列
        # 雑音
        s.Q = np.array(Q) #システム雑音の共分散行列
        s.R = np.array(R) #観測雑音の共分散行列
        
        s.xdim = F.shape[1] #状態ベクトルの次元＝状態推移行列の列数
        s.ydim = H.shape[0] #観測行列の行数
        
        ### KBFの内部状態
        s.xf   = np.array(x0) #濾波推定値
        s.xp   = np.array(x0) #予測推定値
        s.cov0 = cov0         #共分散行列
        s.K    = None         #カルマンゲイン

        ### 定常カルマンフィルタの導出
        RicA = s.F.T
        RicB = s.H.T
        RicQ = s.G.dot(s.Q).dot(s.G.T)
        RicR = s.R
        #リカッチ方程式の解
        s.cov = solve_discrete_are(RicA, RicB, RicQ, RicR) 
        #定常カルマンゲイン
        HSH_R = s.H.dot(s.cov).dot(s.H.T)+s.R
        if HSH_R.ndim > 1:
            pinv = np.linalg.pinv(HSH_R)
        else:
            pinv = 1.0/HSH_R
        s.K   = s.cov.dot(s.H.T).dot(pinv) 
        print('Steady-state Kalman gain =\n', s.K)

    def recursion(s, yt, xp):

        # Filtering: xf ... x_t/t
        xf = xp + s.K.dot( yt - s.H.dot(xp) )   

        # Prediction: xp ... x_t+1/t
        xp = s.F.dot(xf)

        return (xf, xp)
    
    ### フィルタリング
    def filtering(s, y):

        s.xf, s.xp = s.recursion(y, s.xp)

    ### 安定判別
    def stability(s):

        stability_matrix = s.F - np.dot(s.K, s.H)
        val,vec = np.linalg.eig( stability_matrix )
        absval = np.abs(val)
        if absval.max() < 1:
            print( 'This filter is stable' )
        else:
            print( 'This filter is unstable' )
        print( '> Eigenvalues:' )
        for v in val:
            print( '> ' + str(v) )
        print( '> Their absolute values:' )
        for av in absval:
            print( '> ' + str(av) )
