import numpy as np
from scipy.integrate import ode
from scipy.linalg import solve_continuous_are

class class_ssKBF: # Steady-state Kalman-Bucy filter

    def __init__(s, A, D, C, Q, R, x0, cov0=None, CL=None, t0=0.0):
        
        ### システム行列
        s.A = np.array(A) #状態行列
        s.D = np.array(D) #駆動行列
        s.C = np.array(C) #観測行列
        # 雑音
        s.Q = np.array(Q) #システム雑音の共分散
        s.R = np.array(R) #観測雑音の共分散
        
        ### システムのサイズ
        s.xdim   = A.shape[1]   #状態の次元＝Aの列数
        s.ydim   = C.shape[0]   #観測の次元＝Cの行数
        s.covdim = s.xdim**2 #本来は三角成分だけでよいが手抜き

        ### LQG制御用の閉ループ行列(デフォルトはA)
        if CL is None:
            s.CL = s.A
        else:
            s.CL = CL.copy()
        
        ### KBFの初期値
        s.t    = 0.0
        s.xf   = np.array(x0) #濾波推定値
        s.cov0 = cov0         #共分散行列
        s.K    = None         #カルマンゲイン

        ### 定常カルマンフィルタの導出
        RicA = s.A.T
        RicB = s.C.T
        RicQ = s.D.dot(s.Q).dot(s.D.T)
        RicR = s.R
        #リカッチ方程式の解
        s.cov = solve_continuous_are(RicA, RicB, RicQ, RicR) 
        X0    = s.xf
        #定常カルマンゲイン
        s.K   = s.cov.dot(s.C.T).dot(np.linalg.pinv(s.R)) 
        print('Steady-state Kalman gain =\n', s.K)

        ### 常微分方程式のソルバー
        s.solver = ode(s.KBF_ode).set_integrator('dopri5')
        s.solver.set_initial_value( X0, t0 )

    ### KBFを更新する常微分方程式(定常カルマンフィルタの場合)
    def KBF_ode(s, t, x, y):
        
        dx = s.CL.dot(x) + s.K.dot(y - s.C.dot(x))
        
        return dx

    ### フィルタリング
    def filtering(s, y, dt):
        s.solver.set_f_params( y )
        s.solver.integrate(s.solver.t + dt)

        s.xf = s.solver.y #solver.y は s.xf
        s.t = s.solver.t

    ### 安定判別
    def stability(s):
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
