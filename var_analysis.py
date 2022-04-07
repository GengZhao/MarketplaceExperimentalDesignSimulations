#!/usr/bin/env python

import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from Marketplace import Marketplace
from common import *

if __name__ == '__main__':

    #                                                                  n, n_iters,   GTE,     rho,  a_l/a_c steps
    infile1 = open('lrcr_vary_design-2021-04-04_21:58:37.txt', 'r') # 1M, 10k avg, 0.02 GTE, rho=1, LRCR (.1, ..., .9)
    infile2 = open('lrcr_vary_design-2021-04-04_14:25:48.txt', 'r') # 1M, 10k avg, 0.05 GTE, rho=1, LRCR (.1, ..., .9)
    infile3 = open('lrcr_vary_design-2021-04-04_21:59:48.txt', 'r') # 1M, 1k avg, 0.1 GTE, rho=1, LRCR (.1, ..., .9)

    plt.rc('figure', figsize=[6.4, 4.0])
    lss = ['solid', 'dashed', 'dotted']

    idx = 0
    k_samples = 2000
    for infile in [infile1, infile2, infile3]:

        cr_bias_obs = []
        cr_nl_bias_obs = []
        lr_bias_obs = []
        lr_nl_bias_obs = []
        '''
        tsr_bias_obs = []
        tsris_bias_obs = {}
        '''
        cr_std_obs = []
        cr_nl_std_obs = []
        lr_std_obs = []
        lr_nl_std_obs = []
        '''
        tsr_std_obs = []
        tsris_std_obs = {}
        '''

        cr_q01s = []
        cr_q11s = []
        lr_q10s = []
        lr_q11s = []

        a_ls = []
        a_cs = []
        gtes = []

        for l in infile:
            n, m, phi_0, phi_1, rho, a_l, a_c = l.strip().split(' ')
            n = int(n)
            m = int(m)
            a_l = float(a_l)
            a_c = float(a_c)
            # print('=> a_l & a_c:', a_l, '&', a_c)
            phi_0 = float(phi_0)
            phi_1 = float(phi_1)
            rho = float(rho)
            mp = Marketplace(n = n, m = m, phi_0 = phi_0, phi_1 = phi_1)
            gte = mp.GTE
            a_ls.append(a_l)
            a_cs.append(a_c)
            gtes.append(gte)

            # ks = get_ks(infile)
            n_iters = get_int(infile)

            if a_c == 1.0:
                # Listing-side randomization
                # lr, lr_nl = mp.LR_bias_n_times(n_times)
                lr = get_N_floats(n_iters, infile)
                lr_nl = get_N_floats(n_iters, infile)
                lr_q10s.append(get_N_floats(n_iters, infile))
                lr_q11s.append(get_N_floats(n_iters, infile))
                print('LR bias:', np.mean(lr) - gte, '+/-', np.std(lr))
                # print('LR NL bias:', np.mean(lr_nl) - gte, '+/-', np.std(lr_nl))
                lr_bias_obs.append(np.mean(lr[:k_samples]) - gte)
                lr_nl_bias_obs.append(np.mean(lr_nl) - gte)
                lr_std_obs.append(np.std(lr[:k_samples]))
                lr_nl_std_obs.append(np.std(lr_nl))
            elif a_l == 1.0:
                # Customer-side randomization
                # cr, cr_nl = mp.CR_bias_n_times(n_times)
                cr = get_N_floats(n_iters, infile)
                cr_nl = get_N_floats(n_iters, infile)
                cr_q01s.append(get_N_floats(n_iters, infile))
                cr_q11s.append(get_N_floats(n_iters, infile))
                print('CR bias:', np.mean(cr) - gte, '+/-', np.std(cr))
                # print('CR NL:', np.mean(cr_nl) - gte, '+/-', np.std(cr_nl))
                cr_bias_obs.append(np.mean(cr[:k_samples]) - gte)
                cr_nl_bias_obs.append(np.mean(cr_nl) - gte)
                cr_std_obs.append(np.std(cr[:k_samples]))
                cr_nl_std_obs.append(np.std(cr_nl))

        infile.close()

        a_ls = np.array(a_ls)
        a_cs = np.array(a_cs)
        n_a_ls = int(np.round(a_ls.max() / a_ls.min()))
        n_a_cs = int(np.round(a_cs.max() / a_cs.min()))

        plt.plot(a_cs[:(len(a_cs)/2)], np.array(cr_std_obs) / gte, color=colorWheel[0], ls=lss[idx])
        plt.plot(a_cs[:(len(a_cs)/2)], np.array(lr_std_obs) / gte, color=colorWheel[1], ls=lss[idx])
        idx += 1

    xl = plt.xlim()
    yl = plt.ylim()
    series = []
    series.append(plt.plot([-2, -1], [-2,-1], color=colorWheel[0])[0])
    series.append(plt.plot([-2, -1], [-2,-1], color=colorWheel[1])[0])
    series.append(plt.plot([-2, -1], [-2,-1], ls=lss[0], color='gray')[0])
    series.append(plt.plot([-2, -1], [-2,-1], ls=lss[1], color='gray')[0])
    series.append(plt.plot([-2, -1], [-2,-1], ls=lss[2], color='gray')[0])

    plt.xlim(xl)
    # plt.ylim(yl)
    plt.ylim([0.001, 0.016])
    legend1 = plt.legend(series[:2], ['CR', 'LR'], title=r'\textbf{Design}', loc='lower right')
    plt.legend(series[2:], [r'$22\%$', r'$25\%$', r'$30\%$'], title=r'\textbf{GT booking percentage}', loc='upper left')
    plt.gca().add_artist(legend1)

    plt.xlabel(r'Treatment allocation')
    plt.ylabel('SD/GTE')

    plt.tight_layout()
    sns.despine()
    plt.show()


    infile1 = open('lrcr_vary_design-2021-04-04_21:58:37.txt', 'r') # 1M, 10k avg, 0.02 GTE, rho=1, LRCR (.1, ..., .9)
    infile2 = open('lrcr_vary_design-2021-04-04_14:25:48.txt', 'r') # 1M, 10k avg, 0.05 GTE, rho=1, LRCR (.1, ..., .9)
    infile3 = open('lrcr_vary_design-2021-04-04_21:59:48.txt', 'r') # 1M, 1k avg, 0.1 GTE, rho=1, LRCR (.1, ..., .9)

    # infile1 = open('lrcr_vary_design-2021-02-09_10:56:03.txt', 'r') # 1M, 10k avg, 0.02 GTE, rho=1, LRCR (.1, ..., .9)
    # infile2 = open('lrcr_vary_design-2021-02-12_18:50:35.txt', 'r') # 1M, 10k avg, 0.05 GTE, rho=1, LRCR (.1, ..., .9)
    # infile3 = open('lrcr_vary_design-2021-02-12_18:49:48.txt', 'r') # 1M, 1k avg, 0.1 GTE, rho=1, LRCR (.1, ..., .9)

    idx = 0
    k_samples = 2000
    for infile, rates in zip([infile1, infile2, infile3], [[0.2, 0.22], [0.2, 0.25], [0.2, 0.3]]):

        conrate, tilde_conrate = rates

        cr_bias_obs = []
        cr_nl_bias_obs = []
        lr_bias_obs = []
        lr_nl_bias_obs = []
        '''
        tsr_bias_obs = []
        tsris_bias_obs = {}
        '''
        cr_std_obs = []
        cr_nl_std_obs = []
        lr_std_obs = []
        lr_nl_std_obs = []
        '''
        tsr_std_obs = []
        tsris_std_obs = {}
        '''

        cr_q01s = []
        cr_q11s = []
        lr_q10s = []
        lr_q11s = []

        a_ls = []
        a_cs = []
        gtes = []

        ns = []

        for l in infile:
            n, m, phi_0, phi_1, rho, a_l, a_c = l.strip().split(' ')
            n = int(n)
            m = int(m)
            ns.append(n)
            a_l = float(a_l)
            a_c = float(a_c)
            # print('=> a_l & a_c:', a_l, '&', a_c)
            phi_0 = float(phi_0)
            phi_1 = float(phi_1)
            rho = float(rho)
            mp = Marketplace(n = n, m = m, phi_0 = phi_0, phi_1 = phi_1)
            gte = mp.GTE
            a_ls.append(a_l)
            a_cs.append(a_c)
            gtes.append(gte)

            # ks = get_ks(infile)
            n_iters = get_int(infile)

            if a_c == 1.0:
                # Listing-side randomization
                # lr, lr_nl = mp.LR_bias_n_times(n_times)
                lr = get_N_floats(n_iters, infile)
                lr_nl = get_N_floats(n_iters, infile)
                lr_q10s.append(get_N_floats(n_iters, infile))
                lr_q11s.append(get_N_floats(n_iters, infile))
                print('LR bias:', np.mean(lr) - gte, '+/-', np.std(lr))
                # print('LR NL bias:', np.mean(lr_nl) - gte, '+/-', np.std(lr_nl))
                lr_bias_obs.append(np.mean(lr[:k_samples]) - gte)
                lr_nl_bias_obs.append(np.mean(lr_nl) - gte)
                lr_std_obs.append(np.std(lr[:k_samples]))
                lr_nl_std_obs.append(np.std(lr_nl))
            elif a_l == 1.0:
                # Customer-side randomization
                # cr, cr_nl = mp.CR_bias_n_times(n_times)
                cr = get_N_floats(n_iters, infile)
                cr_nl = get_N_floats(n_iters, infile)
                cr_q01s.append(get_N_floats(n_iters, infile))
                cr_q11s.append(get_N_floats(n_iters, infile))
                print('CR bias:', np.mean(cr) - gte, '+/-', np.std(cr))
                # print('CR NL:', np.mean(cr_nl) - gte, '+/-', np.std(cr_nl))
                cr_bias_obs.append(np.mean(cr[:k_samples]) - gte)
                cr_nl_bias_obs.append(np.mean(cr_nl) - gte)
                cr_std_obs.append(np.std(cr[:k_samples]))
                cr_nl_std_obs.append(np.std(cr_nl))

        infile.close()

        a_ls = np.array(a_ls)
        a_cs = np.array(a_cs)
        n_a_ls = int(np.round(a_ls.max() / a_ls.min()))
        n_a_cs = int(np.round(a_cs.max() / a_cs.min()))

        plt.plot(a_cs[:(len(a_cs)/2)], np.sqrt(finite_var_cr(conrate, tilde_conrate, 1.0, a_cs[:(len(a_cs)/2)], ns[:(len(a_cs)/2)])) / gte, color=colorWheel[0], ls=lss[idx])
        plt.plot(a_cs[:(len(a_cs)/2)], np.sqrt(finite_var_lr(conrate, tilde_conrate, 1.0, a_cs[:(len(a_cs)/2)], ns[:(len(a_cs)/2)])) / gte, color=colorWheel[1], ls=lss[idx])
        idx += 1

    xl = plt.xlim()
    yl = plt.ylim()
    series = []
    series.append(plt.plot([-2, -1], [-2,-1], color=colorWheel[0])[0])
    series.append(plt.plot([-2, -1], [-2,-1], color=colorWheel[1])[0])
    series.append(plt.plot([-2, -1], [-2,-1], ls=lss[0], color='gray')[0])
    series.append(plt.plot([-2, -1], [-2,-1], ls=lss[1], color='gray')[0])
    series.append(plt.plot([-2, -1], [-2,-1], ls=lss[2], color='gray')[0])

    plt.xlim(xl)
    # plt.ylim(yl)
    plt.ylim([0.001, 0.016])
    legend1 = plt.legend(series[:2], ['CR', 'LR'], title=r'\textbf{Design}', loc='lower right')
    plt.legend(series[2:], [r'$22\%$', r'$25\%$', r'$30\%$'], title=r'\textbf{GT booking percentage}', loc='upper left')
    plt.gca().add_artist(legend1)

    plt.xlabel(r'Treatment allocation')
    plt.ylabel('SD/GTE')

    plt.tight_layout()
    sns.despine()
    plt.show()


    '''
    bias_series = [tsr_bias_obs] + [tsris_bias_obs[k] for k in ks]
    bias_series = [np.array(serie).reshape((n_a_ls, n_a_cs)) for serie in bias_series]
    std_series = [tsr_std_obs] + [tsris_std_obs[k] for k in ks]
    std_series = [np.array(serie).reshape((n_a_ls, n_a_cs)) for serie in std_series]
    rmse_series = [np.sqrt(b ** 2 + s ** 2) for b,s in zip(bias_series, std_series)]

    min_all = np.amin([bias_series, std_series, rmse_series])
    max_all = np.amax([bias_series, std_series, rmse_series])

    plt.rc('figure', figsize=[16.8, 4.8])

    plt.subplot(1, 3, 1)
    plt.imshow(bias_series[2], norm=LogNorm())
    plt.xticks(np.arange(n_a_ls), ['{:.2f}'.format(x) for x in np.linspace(a_ls.min(), a_ls.max(), n_a_ls)], rotation=90)
    plt.yticks(np.arange(n_a_cs), ['{:.2f}'.format(y) for y in np.linspace(a_cs.min(), a_cs.max(), n_a_cs)])
    plt.xlabel('a_l')
    plt.ylabel('a_c')
    plt.title(labels[2] + ' bias')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(std_series[2], vmax=max_all, norm=LogNorm())
    plt.xticks(np.arange(n_a_ls), ['{:.2f}'.format(x) for x in np.linspace(a_ls.min(), a_ls.max(), n_a_ls)], rotation=90)
    plt.yticks(np.arange(n_a_cs), ['{:.2f}'.format(y) for y in np.linspace(a_cs.min(), a_cs.max(), n_a_cs)])
    plt.xlabel('a_l')
    plt.ylabel('a_c')
    plt.title(labels[2] + ' SD')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(rmse_series[2], vmax=max_all, norm=LogNorm())
    plt.xticks(np.arange(n_a_ls), ['{:.2f}'.format(x) for x in np.linspace(a_ls.min(), a_ls.max(), n_a_ls)], rotation=90)
    plt.yticks(np.arange(n_a_cs), ['{:.2f}'.format(y) for y in np.linspace(a_cs.min(), a_cs.max(), n_a_cs)])
    plt.xlabel('a_l')
    plt.ylabel('a_c')
    plt.title(labels[2] + ' RMSE')
    plt.colorbar()

    plt.show()

    plt.rc('figure', figsize=[16.8, 12])
    n_estimators = len(labels)
    for i in range(n_estimators):
        plt.subplot(3, n_estimators, i + 1)
        plt.imshow(bias_series[i], vmin=min_all, vmax=max_all, norm=LogNorm())
        plt.xticks(np.arange(n_a_ls), ['{:.2f}'.format(x) for x in np.linspace(a_ls.min(), a_ls.max(), n_a_ls)], rotation=90)
        plt.yticks(np.arange(n_a_cs), ['{:.2f}'.format(y) for y in np.linspace(a_cs.min(), a_cs.max(), n_a_cs)])
        plt.xlabel('a_l')
        plt.ylabel('a_c')
        plt.title(labels[i] + ' bias')
        plt.colorbar()

        plt.subplot(3, n_estimators, i + n_estimators + 1)
        plt.imshow(std_series[i], vmin=min_all, vmax=max_all, norm=LogNorm())
        plt.xticks(np.arange(n_a_ls), ['{:.2f}'.format(x) for x in np.linspace(a_ls.min(), a_ls.max(), n_a_ls)], rotation=90)
        plt.yticks(np.arange(n_a_cs), ['{:.2f}'.format(y) for y in np.linspace(a_cs.min(), a_cs.max(), n_a_cs)])
        plt.xlabel('a_l')
        plt.ylabel('a_c')
        plt.title(labels[i] + ' SD')
        plt.colorbar()

        plt.subplot(3, n_estimators, i + 2 * n_estimators + 1)
        plt.imshow(rmse_series[i], vmin=min_all, vmax=max_all, norm=LogNorm())
        plt.xticks(np.arange(n_a_ls), ['{:.2f}'.format(x) for x in np.linspace(a_ls.min(), a_ls.max(), n_a_ls)], rotation=90)
        plt.yticks(np.arange(n_a_cs), ['{:.2f}'.format(y) for y in np.linspace(a_cs.min(), a_cs.max(), n_a_cs)])
        plt.xlabel('a_l')
        plt.ylabel('a_c')
        plt.title(labels[i] + ' RMSE')
        plt.colorbar()

    plt.show()
    
    for i in range(1,n_estimators):
        plt.subplot(3, n_estimators-1, i)
        plt.imshow(bias_series[i]-bias_series[0], vmin=np.amin(np.array(bias_series)-bias_series[0]), vmax=np.amax(np.array(bias_series)-bias_series[0]))
        plt.xticks(np.arange(n_a_ls), ['{:.2f}'.format(x) for x in np.linspace(a_ls.min(), a_ls.max(), n_a_ls)], rotation=90)
        plt.yticks(np.arange(n_a_cs), ['{:.2f}'.format(y) for y in np.linspace(a_cs.min(), a_cs.max(), n_a_cs)])
        plt.xlabel('a_l')
        plt.ylabel('a_c')
        plt.title('Bias: ' + labels[i] + ' - naive TSR')
        plt.colorbar()

        plt.subplot(3, n_estimators-1, i+n_estimators-1)
        plt.imshow(std_series[i]-std_series[0], vmin=np.amin(np.array(std_series)-std_series[0]), vmax=np.amax(np.array(std_series)-std_series[0]))
        plt.xticks(np.arange(n_a_ls), ['{:.2f}'.format(x) for x in np.linspace(a_ls.min(), a_ls.max(), n_a_ls)], rotation=90)
        plt.yticks(np.arange(n_a_cs), ['{:.2f}'.format(y) for y in np.linspace(a_cs.min(), a_cs.max(), n_a_cs)])
        plt.xlabel('a_l')
        plt.ylabel('a_c')
        plt.title('SD: ' + labels[i] + ' - naive TSR')
        plt.colorbar()

        plt.subplot(3, n_estimators-1, i+2*n_estimators-2)
        plt.imshow(rmse_series[i]-rmse_series[0], vmin=np.amin(np.array(rmse_series)-rmse_series[0]), vmax=np.amax(np.array(rmse_series)-rmse_series[0]))
        plt.xticks(np.arange(n_a_ls), ['{:.2f}'.format(x) for x in np.linspace(a_ls.min(), a_ls.max(), n_a_ls)], rotation=90)
        plt.yticks(np.arange(n_a_cs), ['{:.2f}'.format(y) for y in np.linspace(a_cs.min(), a_cs.max(), n_a_cs)])
        plt.xlabel('a_l')
        plt.ylabel('a_c')
        plt.title('RMSE: ' + labels[i] + ' - naive TSR')
        plt.colorbar()

    plt.show()
    '''

    '''
    legend_labels = ['LR', 'LRNL', 'CR', 'CRNL', 'TSR'] + ['TSRI-' + str(k) for k in ks]
    # legend_labels = ['LR', 'CR', 'TSR']
    bias_series = [lr_bias_obs, lr_nl_bias_obs, cr_bias_obs, cr_nl_bias_obs, tsr_bias_obs] + tsris_bias_obs.values()
    # bias_series = [lr_bias_obs, cr_bias_obs, tsr_bias_obs]
    std_series = [lr_std_obs, lr_nl_std_obs, cr_std_obs, cr_nl_std_obs, tsr_std_obs] + tsris_std_obs.values()
    # std_series = [lr_std_obs, cr_std_obs, tsr_std_obs]
    bias_series = [np.array(serie) for serie in bias_series]
    std_series = [np.array(serie) for serie in std_series]
    mse_series = [np.sqrt(b ** 2 + s ** 2) for b,s in zip(bias_series, std_series)]

    plt.rc('figure', figsize=[10, 12])

    # Relative bias
    plt.subplot(3, 2, 1)
    series = []
    for serie in bias_series:
        # series.append(plt.plot(rhos, np.abs(serie) / np.array(gtes))[0])
        series.append(plt.plot(ns, np.abs(serie) / np.array(gtes))[0])

    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('|bias/GTE|')
    plt.legend(series, legend_labels)

    # Normalized bias
    plt.subplot(3, 2, 2)
    series = []
    for serie in bias_series:
        # series.append(plt.plot(rhos, np.abs(serie) / np.minimum(1.0, rhos))[0])
        series.append(plt.plot(ns, np.abs(serie) / np.minimum(1.0, rhos))[0])

    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('Normalized bias')
    plt.legend(series, legend_labels)

    # Relative scaled sqrt variance
    plt.subplot(3, 2, 3)
    series = []
    for serie in std_series:
        # series.append(plt.plot(rhos, np.array(serie) / np.array(gtes))[0])
        series.append(plt.plot(ns, np.sqrt(ns) * np.array(serie) / np.array(gtes))[0])

    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('sqrt(n) Stdev/GTE')
    plt.legend(series, legend_labels)

    # Normalized scaled sqrt variance
    plt.subplot(3, 2, 4)
    series = []
    for serie in std_series:
        # series.append(plt.plot(rhos, np.array(serie) / np.minimum(1.0, rhos))[0])
        series.append(plt.plot(ns, np.sqrt(ns) * np.array(serie) / np.minimum(1.0, rhos))[0])

    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('sqrt(n) normalized stdev')
    plt.legend(series, legend_labels)

    # Relative MSE
    plt.subplot(3, 2, 5)
    series = []
    for serie in mse_series:
        # series.append(plt.plot(rhos, np.abs(serie) / np.array(gtes))[0])
        series.append(plt.plot(ns, np.abs(serie) / np.array(gtes))[0])

    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('MSE/GTE')
    plt.legend(series, legend_labels)

    # Normalized MSE
    plt.subplot(3, 2, 6)
    series = []
    for serie in mse_series:
        # series.append(plt.plot(rhos, np.abs(serie) / np.minimum(1.0, rhos))[0])
        series.append(plt.plot(ns, np.abs(serie) / np.minimum(1.0, rhos))[0])

    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('Normalized MSE')
    plt.legend(series, legend_labels)

    plt.show()
    '''

