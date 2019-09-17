import numpy as np
import matplotlib.pyplot as plt

### デフォルト値
default_values = ( #Q, R, C, D, x0, dt, tn, p_list
    np.diag([0.0001,0.0001]), #Q:  システム雑音の共分散
    np.array([[0.0001]]),     #R:  観測雑音の共分散
    np.array([[1, 0]]),       #C:  変位のみ
    np.eye(2),                #D:  駆動行列
    np.array([0.0, 0.0]),     #x0: 初期値
    0.02,                     #dt: 時間ステップ
    2000,                     #tn: 時系列長
    [1, 0.2, 1],              #p_list: k, c, a
)

### 外部励振
def Forcing(t):
    return np.sin(1.5*t)

### 拡大系の導出
def get_extended_system(x0, Q, D, C, A=None, Qval=1.2):
        
    ex0 = np.append(x0, [0.0])
    # 拡大系
    if A is not None:
        eA = np.pad(A, (0,1), 'constant') #Aを1行1列拡大して0で埋める
        eA[-2,-1] = 1 #必要箇所に1を代入する
    else:
        eA = None
    # 推定パラメータにもシステム雑音を仮定（無いと推定パラメータが動かなくなる）
    eD = np.pad(D, (0,1), 'constant')
    eD[-1,-1] = 1
    # 推定パラメータは観測できないとする
    eC = np.pad(C, (0,1), 'constant')
    eC = np.delete(eC, axis=0, obj=-1)
    # 推定パラメータ用のシステム雑音強度を追加
    eQ = np.pad(Q, (0,1), 'constant')
    eQ[-1,-1] = Qval    #あまり小さいと収束が遅いが，大きすぎても効果なし．

    print('Extended x0 =\n',ex0)
    print('Extended A =\n',eA)
    print('Extended D =\n',eD)
    print('Extended C =\n',eC)
    print('Extended Q =\n',eQ)
        
    if A is not None:
        return (ex0, eQ, eD, eC, eA)
    else:
        return (ex0, eQ, eD, eC)


### サンプルデータの取得
from class_SDE import * #実行フォルダにclass_SDE.pyを置いてインポートします．

class model_1dof_tv(class_SDE): #class_SDEの子クラス, 1自由度振動系，時変パラメータ

    def __init__(s, p_idx, p1, p2, t1=None):
        
        Q, R, C, D, x0, dt, tn, p_list = default_values
        s.C = C
        s.D = D
        s.tn = tn
        s.p_list = p_list.copy()
        
        xdim = 2               #状態の次元
        ydim = s.C.shape[0] #観測の次元＝Cの行数
        t0 = 0.0
        
        super().__init__(xdim, ydim, Q, R)
        s.setup(x0, dt, t0)

        # 時刻t1で変化するパラメータ
        s.p_idx  = p_idx   
        s.p1 = p1
        s.p2 = p2
        if t1 is None:
            s.t1=(s.tn*s.dt)/2
        else:
            s.t1 = t1
        
    ### 状態方程式の定義(必須)
    def StateEqn(s, t, x):
        
        if t<s.t1:
            s.p_list[s.p_idx] = s.p1
        else:
            s.p_list[s.p_idx] = s.p2
               
        k, c, a = s.p_list
        Dw = np.ravel(s.D.dot(s.w))
        
        s.dx[0] = x[1] + Dw[0]
        s.dx[1] = - k*x[0] - c*x[1] + a*Forcing(t) + Dw[1]
        
        return s.dx
    
    ### 観測方程式の定義(必須)
    def OutputEqn(s, x):
        return s.C.dot(x)
    
    ### サンプルデータの取得
    def get_sample_path(s): 
        s.tt, s.xx, s.yy = super().get_sample_path(s.tn)
        s.param = np.zeros_like(s.tt)
        for i, t in enumerate(s.tt):
            if t<s.t1:
                s.param[i] = s.p1
            else:
                s.param[i] = s.p2

### プロット
def plot(cls, param_label='Parameter'):

    fig, ax = plt.subplots(3, 1, figsize=(4,4))

    lb_exa = 'Exact'
    ls_exa = ':'
    lc_exa = 'black'
    lw_exa = 1.5
    lw_est = 1.5

    ax[0].plot(cls.tt, cls.xx[:,0],  ls_exa, label=lb_exa, color=lc_exa, linewidth=lw_exa )
    ax[0].plot(cls.tt, cls.xxf[:,0], '-', label='Estimated', color='black',   linewidth=lw_est )
    ax[0].set_ylabel('$x_1$', fontsize=12)
    ax[0].legend(bbox_to_anchor=(1.0, 0.85), loc='lower right', ncol=2)

    ax[1].plot(cls.tt, cls.xx[:,1], ls_exa, label=None, color=lc_exa,  linewidth=lw_exa )
    ax[1].plot(cls.tt, cls.xxf[:,1],   '-', label=None, color='black', linewidth=lw_est )
    ax[1].set_ylabel('$x_2$', fontsize=12)

    for i in range(2):
        plt.setp(ax[i].get_xticklabels(),visible=False)
        ax[i].grid(); 

    ax[2].plot(cls.tt, cls.param, ls_exa, label=lb_exa,  color=lc_exa,  linewidth=lw_exa )
    ax[2].plot(cls.tt, cls.xxf[:,2], '-', label='Estimated', color='black', linewidth=lw_est )
    ax[2].set_ylabel(param_label, fontsize=12)
    ax[2].set_xlabel('$t$', fontsize=12)
    ax[2].grid(); 
        
    plt.tight_layout()

def save(filename):
    plt.savefig(filename, bbox_inches="tight")
