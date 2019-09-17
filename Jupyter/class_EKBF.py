import numpy as np
from scipy.integrate import ode

class class_EKBF: #拡張Kalman-Bucyフィルタ(Extended Kalman-Bucy filter)

    def __init__(s, xdim, ydim, D, Q, R, x0, cov0=None, t0=0.0):
        
        s.xdim = xdim #状態ベクトルの次元
        s.ydim = ydim #出力ベクトルの次元

        ### システム行列
        ### 状態行列 s.A は推定時に代入される
        ### 観測行列 s.C は推定時に代入される
        s.D = np.array(D) #駆動行列
        # 雑音
        s.Q = np.array(Q) #システム雑音の共分散
        s.R = np.array(R) #観測雑音の共分散
        
        ### 共分散行列のベクトル化の次元
        s.covdim = s.xdim**2 #本来は三角成分だけでよいが手抜き

        ### フィルタの初期値
        s.t    = t0           #初期時刻
        s.xf   = np.array(x0) #濾波推定値
        s.K    = None         #カルマンゲイン
        #共分散行列とその初期値
        if cov0 is not None:
            s.cov = np.array(cov0)
        else:
            s.cov = np.zeros((s.xdim,s.xdim))

        #期待値と共分散行列を1列に並べた状態ベクトル
        X0 = s.xcov2vec(s.xf, s.cov)

        ### 常微分方程式のソルバー
        s.solver = ode(s.ode_func).set_integrator('dopri5')
        s.solver.set_initial_value( X0, t0 )

    def vec2xcov(self, vec):
        x   = vec[:self.xdim]
        cov = vec[self.xdim:].reshape(self.xdim,self.xdim)
        return (x,cov)

    def xcov2vec(self, x, cov):
        vec   = np.zeros(self.xdim + self.covdim)
        vec[:self.xdim] = x
        vec[self.xdim:] = np.ravel(cov)
        return vec
    
    ### KBFを更新する常微分方程式
    def ode_func(s, t, X, y):

        x,cov = s.vec2xcov(X) #期待値と共分散行列の切り分け

        C = s.C_jac(x,t)                               #拡張C_jac
        s.C = C
        
        # カルマンゲイン
        K = cov.dot(C.T).dot(np.linalg.pinv(s.R))
        s.K = K
        
        # 期待値の常微分方程式
        dx = s.A_func(x, t) + s.K.dot(y - s.C_func(x, t)) #拡張A_func, C_func

        # 共分散の常微分方程式
        A = s.A_jac(x,t)                               #拡張A_jac
        s.A = A

        dcov = A.dot(cov) + cov.dot(A.T) \
              + s.D.dot(s.Q).dot(s.D.T) - s.K.dot(C).dot(cov)
        
        dX = s.xcov2vec(dx,dcov) #ベクトル化
        
        return dX

    def filtering(s, y, dt, t=None ): 

        s.solver.set_f_params( y )

        if t is not None:
            s.solver.set_initial_value( s.solver.y, t )

        s.solver.integrate(s.solver.t + dt)

        s.xf, s.cov = s.vec2xcov(s.solver.y) 
        s.t = s.solver.t

    def stability(s): #参考程度だが一応残しておく
        stability_matrix = s.A - np.dot(s.K, s.C)
        val,vec = np.linalg.eig( stability_matrix )
        realval = np.real(val)
        if realval.max() < 0:
            print( 'This filter is stable' )
        else:
            print( 'This filter is unstable' )
        print( '> Eigenvalues:' )
        for v in val:
            print( '> ' + str(v) )
