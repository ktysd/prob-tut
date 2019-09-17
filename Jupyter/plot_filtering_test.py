import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_filtering_test( cls ):
    fig = plt.figure(figsize=(5, 3.5))
    gs = gridspec.GridSpec(2,2)
    ax1 = fig.add_subplot(gs[0,:])
    ax1.plot(cls.tt, cls.xx[:,1], '--', color='gray', linewidth=2, alpha=0.6, label='Exact')
    ax1.plot(cls.tt, cls.yy,      'k-', linewidth=0.7, alpha=1.0, label='Noisy measurement')
    ax1.set_xlim([cls.tt.min(),cls.tt.max()])
    ax1.set_xlabel('$t$',fontsize=12)
    ax1.set_ylabel('$y$',fontsize=12)
    ax1.legend(bbox_to_anchor=(0.5, 1.0), loc='lower center')
    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(cls.xx[:,0], cls.xx[:,1], '--', color='gray', linewidth=2, alpha=0.6)
    if cls.xxf is not None:
        ax2.plot(cls.xxf[:,0], cls.xxf[:,1], 'k-', linewidth=0.7, alpha=1.0, label='KF filtered')
    ax2.set_xlabel('$x_1$',fontsize=12)
    ax2.set_ylabel('$x_2$',fontsize=12)
    ax2.legend(bbox_to_anchor=(0.5, 1.0), loc='lower center')
    if cls.xxni is not None:
        ax3 = fig.add_subplot(gs[1,1])
        ax3.plot(cls.xx[:,0], cls.xx[:,1], '--', color='gray', linewidth=2, alpha=0.6)
        ax3.plot(cls.xxni[:,0], cls.xxni[:,1], 'k-', linewidth=0.7, alpha=1.0, label='NI estimated')
        ax3.set_xlabel('$x_1$',fontsize=12)
        ax3.set_ylabel('$x_2$',fontsize=12)
        ax3.legend()
        ax3.legend(bbox_to_anchor=(0.5, 1.0), loc='lower center')

    plt.tight_layout()
