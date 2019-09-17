import numpy as np
from scipy.integrate import ode

class class_SDE:
    # dx/dt = state_eqn(t, x)
    # y = output_eqn(x) + v

    def __init__(self, xdim, ydim, Q, R):

        ### システムのサイズ
        self.xdim = xdim #状態の次元
        self.ydim = ydim #観測の次元

        ### 雑音
        self.Q    = np.array(Q)         #システム雑音の共分散
        self.wdim = self.Q.shape[0]     #システム雑音の次元
        self.w    = np.zeros(self.wdim) #システム雑音ベクトル
        self.wav  = np.zeros(self.wdim) #システム雑音の平均
        self.R    = np.array(R)         #観測雑音の共分散
        self.vdim = self.R.shape[0]     #観測雑音の次元
        self.v    = np.zeros(self.vdim) #観測雑音ベクトル
        self.vav  = np.zeros(self.vdim) #観測雑音の平均

    def setup(s, x0, dt, t0=0.0, solver_type='dopri5'):
        s.x0 = np.array(x0) #初期状態ベクトル
        s.dx = np.zeros_like(s.x0) #状態ベクトルの時間微分
        s.dt = dt           #積分ステップ 
        s.t0 = t0           #初期時刻
        ### ソルバの設定
        s.solver = ode(s.StateEqn).set_integrator(solver_type)
        s.solver.set_initial_value( s.x0, s.t0 )
        ### クラス変数をソルバと同期
        s.t = s.solver.t #クラス変数(t,x)はsolver内の(x,y)に対応
        s.x = s.solver.y     #状態ベクトルの初期値
        s.y = s.get_output() #観測ベクトルの初期値

    def StateEqn(s, t, x): 
        # dx/dt = StateEqn(t, x)
        # 関数内でシステム雑音 s.w, 一時保持用の s.dx が使える．
        ### dummy ###
        s.dx = x
        return s.dx

    def OutputEqn(s, x): #dx/dt = StateEqn(t, x)
        # y = output_func(x) + v
        # 観測雑音 v はget_output()で自動的に加算される 
        ### dummy ###
        y = x
        return y

    def get_output(s): 
        ### 雑音の更新
        s.update_v()
        return s.OutputEqn(s.x) + s.v

    def update_w(s):
        if s.wdim == 1:
            s.w = np.sqrt(s.Q[0]) * np.random.randn() #正規乱数
        else:
            s.w = np.random.multivariate_normal(s.wav, s.Q)
        
    def update_v(s):
        if s.vdim == 1:
            s.v = np.sqrt(s.R[0]) * np.random.randn() #正規乱数
        else:
            s.v = np.random.multivariate_normal(s.vav, s.R)

    def propagator(s, x0, t0):    
        ### 雑音の更新
        s.update_w()

        # 算法12.1にあるDの補正と同じことを，wの補正でしている．
        # 砂原「確率システム理論」ISBN:4-88552-028-2, 89頁など
        inv_sqrt_dt = 1.0/np.sqrt(s.dt)
        s.w *= inv_sqrt_dt

        s.solver.set_initial_value( x0, t0 )
        s.solver.integrate(s.solver.t + s.dt)

        return (s.solver.y, s.solver.t) #本クラスの(t,x)はsolverの(t,y)
        
    def solve(s):
        s.x, s.t = s.propagator(s.x, s.t)

    def set_input(s, u):
        s.u = u

    def get_sample_path(s, tn):
        tt = np.zeros(tn+1)   #時刻の列
        xx = np.zeros((tn+1, s.xdim)) #状態ベクトルの時系列
        yy = np.zeros((tn+1, s.ydim)) #観測ベクトルの時系列

        for i in range(tn+1):
            tt[i] = s.t
            xx[i,:] = s.x
            yy[i] = s.get_output()
            s.solve()

        return tt, xx, yy
