#!/usr/bin/env python
from common import *
import pandas as pd

plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)

def argmin_nvar_cr(conrate, tildeconrate, lmbd):
    u = 0.99
    l = 0.01
    uv = nvar_cr(conrate, tildeconrate, lmbd, u)
    lv = nvar_cr(conrate, tildeconrate, lmbd, l)
    while u - l > 0.001:
        if uv > lv:
            u = 0.6 * u + 0.4 * l
            uv = nvar_cr(conrate, tildeconrate, lmbd, u)
        else:
            l = 0.6 * l + 0.4 * u
            lv = nvar_cr(conrate, tildeconrate, lmbd, l)
    return (u + l) / 2.0

def argmin_nvar_lr(conrate, tildeconrate, lmbd):
    u = 0.99
    l = 0.01
    uv = nvar_lr(conrate, tildeconrate, lmbd, u)
    lv = nvar_lr(conrate, tildeconrate, lmbd, l)
    while u - l > 0.001:
        if uv > lv:
            u = 0.6 * u + 0.4 * l
            uv = nvar_lr(conrate, tildeconrate, lmbd, u)
        else:
            l = 0.6 * l + 0.4 * u
            lv = nvar_lr(conrate, tildeconrate, lmbd, l)
    return (u + l) / 2.0

def nvar_cr_gap(conrate, tildeconrate, lmbd):
    return nvar_cr(conrate, tildeconrate, lmbd, 0.5) / nvar_cr(conrate, tildeconrate, lmbd, argmin_nvar_cr(conrate, tildeconrate, lmbd))

def nvar_lr_gap(conrate, tildeconrate, lmbd):
    return nvar_lr(conrate, tildeconrate, lmbd, 0.5) / nvar_lr(conrate, tildeconrate, lmbd, argmin_nvar_lr(conrate, tildeconrate, lmbd))

lss = ['solid', 'dashed', 'dotted']

if __name__ == '__main__':

    data = pd.DataFrame([[nvar_lr_gap(0.2, 0.2 * tr, lmbd) for tr in np.logspace(0.001, 0.26, 38, base=2)] for lmbd in np.logspace(-1.5, 1.5, 31, base=10)],
            index=["{:.3f}".format(x) for x in np.logspace(-1.5, 1.5, 31, base=10)],
            columns=["{:.3f}".format(x) for x in np.logspace(0.001, 0.26, 38, base=2)])
    sns.heatmap(data,
            cmap="Greys")
    # plt.title('LR Variance Approximation Ratio')
    plt.xlabel(r'$Q_{GT}/Q_{GC}$')
    plt.ylabel(r'$\lambda$')
    plt.tight_layout()
    plt.show()
    data = pd.DataFrame([[nvar_cr_gap(0.2, 0.2 * tr, lmbd) for tr in np.logspace(0.001, 0.26, 38, base=2)] for lmbd in np.logspace(-1.5, 1.5, 31, base=10)],
            index=["{:.3f}".format(x) for x in np.logspace(-1.5, 1.5, 31, base=10)],
            columns=["{:.3f}".format(x) for x in np.logspace(0.001, 0.26, 38, base=2)])
    sns.heatmap(data,
            cmap="Greys")
    # plt.title('CR Variance Approximation Ratio')
    plt.xlabel(r'$Q_{GT}/Q_{GC}$')
    plt.ylabel(r'$\lambda$')
    plt.tight_layout()
    plt.show()
    '''
    plt.plot(np.logspace(0.1, 0.6, 11, base=2), [argmin_nvar_cr(0.2, tr * 0.2, 0.1) for tr in np.logspace(0.1, 0.6, 11, base=2)], color=colorWheel[0], ls=lss[0])
    plt.plot(np.logspace(0.1, 0.6, 11, base=2), [argmin_nvar_cr(0.2, tr * 0.2, 1.) for tr in np.logspace(0.1, 0.6, 11, base=2)], color=colorWheel[1], ls=lss[0])
    plt.plot(np.logspace(0.1, 0.6, 11, base=2), [argmin_nvar_cr(0.2, tr * 0.2, 10.) for tr in np.logspace(0.1, 0.6, 11, base=2)], color=colorWheel[2], ls=lss[0])
    plt.plot(np.logspace(0.1, 0.6, 11, base=2), [argmin_nvar_lr(0.2, tr * 0.2, 0.1) for tr in np.logspace(0.1, 0.6, 11, base=2)], color=colorWheel[0], ls=lss[1])
    plt.plot(np.logspace(0.1, 0.6, 11, base=2), [argmin_nvar_lr(0.2, tr * 0.2, 1.) for tr in np.logspace(0.1, 0.6, 11, base=2)], color=colorWheel[1], ls=lss[1])
    plt.plot(np.logspace(0.1, 0.6, 11, base=2), [argmin_nvar_lr(0.2, tr * 0.2, 10.) for tr in np.logspace(0.1, 0.6, 11, base=2)], color=colorWheel[2], ls=lss[1])
    plt.xlabel(r'$Q_{GT} / Q_{GC}$')
    # plt.ylabel(r'$SD^2(0.5) / SD^2(a^*)$')
    plt.ylabel(r'Optimal treatment allocation $a^*$')
    # plt.xscale('log')
    xl = plt.xlim()
    yl = plt.ylim()
    series = []
    series.append(plt.plot([-2, -1], [-2,-1], color=colorWheel[0])[0])
    series.append(plt.plot([-2, -1], [-2,-1], color=colorWheel[1])[0])
    series.append(plt.plot([-2, -1], [-2,-1], color=colorWheel[2])[0])
    series.append(plt.plot([-2, -1], [-2,-1], ls=lss[0], color='gray')[0])
    series.append(plt.plot([-2, -1], [-2,-1], ls=lss[1], color='gray')[0])
    plt.xlim(xl)
    plt.ylim(yl)
    legend1 = plt.legend(series[:3], [r'$0.1$', r'$1$', r'$10$'], title=r'$\lambda$', loc='upper left')
    plt.legend(series[3:], ['CR', 'LR'], title=r'\textbf{Design}', loc='center left')
    plt.gca().add_artist(legend1)
    plt.tight_layout()
    plt.show()

    plt.plot(np.logspace(-1.5, 1.5, 11, base=10), [nvar_cr_gap(0.2, 0.22, lmbd) for lmbd in np.logspace(-1.5, 1.5, 11, base=10)], color=colorWheel[0], ls=lss[0])
    plt.plot(np.logspace(-1.5, 1.5, 11, base=10), [nvar_cr_gap(0.2, 0.25, lmbd) for lmbd in np.logspace(-1.5, 1.5, 11, base=10)], color=colorWheel[1], ls=lss[0])
    plt.plot(np.logspace(-1.5, 1.5, 11, base=10), [nvar_cr_gap(0.2, 0.3, lmbd) for lmbd in np.logspace(-1.5, 1.5, 11, base=10)], color=colorWheel[2], ls=lss[0])
    plt.plot(np.logspace(-1.5, 1.5, 11, base=10), [nvar_lr_gap(0.2, 0.22, lmbd) for lmbd in np.logspace(-1.5, 1.5, 11, base=10)], color=colorWheel[0], ls=lss[1])
    plt.plot(np.logspace(-1.5, 1.5, 11, base=10), [nvar_lr_gap(0.2, 0.25, lmbd) for lmbd in np.logspace(-1.5, 1.5, 11, base=10)], color=colorWheel[1], ls=lss[1])
    plt.plot(np.logspace(-1.5, 1.5, 11, base=10), [nvar_lr_gap(0.2, 0.3, lmbd) for lmbd in np.logspace(-1.5, 1.5, 11, base=10)], color=colorWheel[2], ls=lss[1])
    plt.xlabel(r'$\lambda=M/N$')
    plt.ylabel(r'Optimal treatment allocation $a^*$')
    plt.ylabel(r'$SD^2(0.5) / SD^2(a^*)$')
    # plt.xscale('log')
    xl = plt.xlim()
    yl = plt.ylim()
    series = []
    series.append(plt.plot([-2, -1], [-2,-1], color=colorWheel[0])[0])
    series.append(plt.plot([-2, -1], [-2,-1], color=colorWheel[1])[0])
    series.append(plt.plot([-2, -1], [-2,-1], color=colorWheel[2])[0])
    series.append(plt.plot([-2, -1], [-2,-1], ls=lss[0], color='gray')[0])
    series.append(plt.plot([-2, -1], [-2,-1], ls=lss[1], color='gray')[0])
    plt.xlim(xl)
    plt.ylim(yl)
    legend1 = plt.legend(series[:3], [r'$22\%$', r'$25\%$', r'$30\%$'], title=r'$Q_{GT}$', loc='lower left')
    plt.legend(series[3:], ['CR', 'LR'], title=r'\textbf{Design}', loc='lower center')
    plt.gca().add_artist(legend1)
    plt.tight_layout()
    plt.show()

    lmbds = [0.1, 0.33, 1., 3., 10.]
    plt.rc('figure', figsize=[21.6, 8])

    for idx, lmbd in enumerate(lmbds):

        plt.subplot(2, len(lmbds), idx + 1)
        plt.plot(np.arange(0.05, 0.96, 0.05), np.sqrt([nvar_cr(0.2, 0.22, lmbd, a_c) for a_c in np.arange(0.05, 0.96, 0.05)]))
        plt.plot(np.arange(0.05, 0.96, 0.05), np.sqrt([nvar_cr(0.2, 0.25, lmbd, a_c) for a_c in np.arange(0.05, 0.96, 0.05)]))
        plt.plot(np.arange(0.05, 0.96, 0.05), np.sqrt([nvar_cr(0.2, 0.3, lmbd, a_c) for a_c in np.arange(0.05, 0.96, 0.05)]))
        # plt.yscale('log')
        plt.xlabel(r'$a_C$')
        plt.ylabel(r'$\sqrt{N}\cdot SD$')
        plt.legend([r'$22\%$', r'$25\%$', r'$30\%$'], title=r'$\lambda$')
        plt.title(r'$\lambda={}, Q_{{GC}}=20\%$'.format(lmbd))
        plt.tight_layout()
        # plt.show()

        plt.subplot(2, len(lmbds), idx + len(lmbds) + 1)
        plt.plot(np.arange(0.05, 0.96, 0.05), np.sqrt([nvar_lr(0.2, 0.22, lmbd, a_l) for a_l in np.arange(0.05, 0.96, 0.05)]))
        plt.plot(np.arange(0.05, 0.96, 0.05), np.sqrt([nvar_lr(0.2, 0.25, lmbd, a_l) for a_l in np.arange(0.05, 0.96, 0.05)]))
        plt.plot(np.arange(0.05, 0.96, 0.05), np.sqrt([nvar_lr(0.2, 0.3, lmbd, a_l) for a_l in np.arange(0.05, 0.96, 0.05)]))
        # plt.yscale('log')
        plt.xlabel(r'$a_L$')
        plt.ylabel(r'$\sqrt{N}\cdot SD$')
        plt.legend([r'$22\%$', r'$25\%$', r'$30\%$'], title=r'$\lambda$')
        plt.title(r'$\lambda={}, Q_{{GC}}=20\%$'.format(lmbd))
        plt.tight_layout()

    plt.show()
    '''

