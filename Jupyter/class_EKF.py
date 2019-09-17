import numpy as np
from scipy.linalg import solve_discrete_are

class class_EKF: #拡張カルマンフィルタ(Extended Kalman filter)
    
    def __init__(s, xdim, ydim, G, Q, R, x0, cov0=None):
        
        s.xdim = xdim #状態ベクトルの次元
        s.ydim = ydim #出力ベクトルの次元
        
        ### システム行列
        ### 状態推移行列 s.F は推定時に代入される
        ### 観測行列 s.H は推定時に代入される
        s.G = np.array(G) #駆動行列
        # 雑音
        s.Q = np.array(Q) #システム雑音の共分散行列
        s.R = np.array(R) #観測雑音の共分散行列
        
        ### フィルタの初期値
        s.xf = np.array(x0) #濾波推定値
        s.xp = np.array(x0) #予測推定値
        s.K  = None         #カルマンゲイン
        #共分散行列とその初期値
        if cov0 is not None:
            s.cov    = np.array(cov0)
        else:
            s.cov    = np.zeros((s.xdim,s.xdim))
            
    def recursion(s, yt, xp, cov, t): #yt:観測量, xp=x_t/t-1, cov=S_t/t-1
        
        H = s.H_jac(xp,t)                      #拡張 H_jac
        s.H = H
        
        HSH_R = H.dot(cov).dot(H.T)+s.R
        if HSH_R.ndim > 1:
            pinv = np.linalg.pinv(HSH_R)
        else:
            pinv = 1.0/HSH_R

        # カルマンゲイン
        K = cov.dot(H.T).dot(pinv)
        s.K = K

        # 濾波推定: xf ... x_t/t
        xf = xp + K.dot( yt - s.H_func(xp,t) ) #拡張 H_func 

        # Prediction: xp ... x_t+1/t
        xp = s.F_func(xf,t)                    #拡張 F_func
        
        # Prediction: Sp ... S_t+1/t
        F = s.F_jac(xf,t)                      #拡張 F_jac
        s.F = F
        
        cov = F.dot(cov).dot(F.T) + s.G.dot(s.Q).dot(s.G.T) \
                - F.dot(K).dot(H).dot(cov).dot(s.F.T)
        
        return (xf, xp, cov)
    
    def filtering(s, y, t=0):
        
        s.xf, s.xp, s.cov = s.recursion(y, s.xp, s.cov, t)

    def stability(s):

        stability_matrix = s.F - s.F.dot(s.K).dot(s.H)
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
