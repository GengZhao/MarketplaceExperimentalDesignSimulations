#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from Marketplace import Marketplace
from common import *

if __name__ == '__main__':

    cr_bias_obs_series = []
    # cr_nl_bias_obs = []
    lr_bias_obs_series = []
    # lr_nl_bias_obs = []
    # tsr_bias_obs = []
    # tsris_bias_obs = {}
    cr_std_obs_series = []
    # cr_nl_std_obs = []
    lr_std_obs_series = []
    # lr_nl_std_obs = []
    # tsr_std_obs = []
    # tsris_std_obs = {}

    lr_q10_m = []
    lr_q11_m = []
    cr_q01_m = []
    cr_q11_m = []

    ns = []
    ms = []
    rhos = []
    gtes = []

    # infile = open('eval_estimators-2020-09-20_15:13:14.txt', 'r')
    # infile = open('eval_estimators-2020-09-21_18:18:58.txt', 'r') # 8:1 - 10x2000
    # infile = open('eval_estimators-2020-09-21_22:48:11.txt', 'r') # 8:1 - 10x2000
    # infile = open('eval_estimators-2020-09-21_23:31:55.txt', 'r') # 1:8 - 10x5000
    # infile = open('eval_estimators-2020-09-25_23:24:00.txt', 'r') # 1:1 - 10x2000
    # infile = open('eval_estimators-2020-09-26_00:44:18.txt', 'r') # 1:1 - 10x5000
    # infile = open('eval_estimators-2020-09-26_10:54:24.txt', 'r') # 1:1 - 12x10000
    # infile = open('eval_estimators-2020-09-26_12:02:04.txt', 'r') # 1:1 - 14x10000 larger gap
    # infile = open('eval_estimators-2020-10-22_17:42:57.txt', 'r') # 1:1 - 14x20000 16M- 10-13pct
    # infile = open('eval_estimators-2020-09-26_11:36:14.txt', 'r') # 1:1 - 14x10000
    # infile = open('eval_estimators-2020-09-27_22:04:45.txt', 'r') # 1:1 - 14x10000 large size
    # infile = open('eval_estimators-2020-09-26_16:30:50.txt', 'r') # 8:1 - 14x10000
    # infile = open('eval_estimators-2020-09-27_22:20:40.txt', 'r') # 8:1 - 14x10000 large size
    # infile = open('eval_estimators-2020-09-26_16:54:16.txt', 'r') # 1:8 - 14x10000
    # infile = open('eval_estimators-2020-10-02_15:19:06.txt', 'r') # 1:1 - 4x50000 small market

    # infile = open('eval_estimators-2021-02-01_18:34:06.txt', 'r') # change rho

    # infile1 = open('eval_estimators-2021-02-02_18:07:50.txt', 'r') # m=2^21, n=2^26-2^16, a_L=a_C=0.5
    # infile2 = open('eval_estimators-2021-02-02_18:09:02.txt', 'r') # m=2^21, n=2^26-2^16, a_L=a_C=0.1
    # infile3 = open('eval_estimators-2021-02-02_18:11:08.txt', 'r') # m=2^21, n=2^26-2^16, a_L=a_C=0.9
    # infile4 = open('eval_estimators-2021-02-02_22:12:33.txt', 'r') # m=2^21, n=2^26-2^16, a_L=a_C=0.3
    # infile5 = open('eval_estimators-2021-02-02_22:13:32.txt', 'r') # m=2^21, n=2^26-2^16, a_L=a_C=0.7

    infile1 = open('eval_estimators-2021-02-03_00:22:46.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.5
    infile2 = open('eval_estimators-2021-02-03_00:23:53.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.1
    infile3 = open('eval_estimators-2021-02-03_00:25:51.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.9
    infile4 = open('eval_estimators-2021-02-03_00:27:22.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.3
    infile5 = open('eval_estimators-2021-02-03_00:28:28.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.7

    # infile1 = open('eval_estimators-2021-02-03_10:29:02.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.5
    # infile2 = open('eval_estimators-2021-02-03_12:54:19.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.1
    # infile3 = open('eval_estimators-2021-02-03_12:55:21.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.9
    # infile4 = open('eval_estimators-2021-02-03_12:56:15.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.3
    # infile5 = open('eval_estimators-2021-02-03_12:56:56.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.7

    '''
    # heterogeneous
    infile1 = open('eval_estimators-2021-02-03_14:39:56.txt', 'r') # m=2^20, n=2^24-2^16, a_L=a_C=0.5
    infile2 = open('eval_estimators-2021-02-03_14:48:14.txt', 'r') # m=2^20, n=2^24-2^16, a_L=a_C=0.1
    infile3 = open('eval_estimators-2021-02-03_14:56:01.txt', 'r') # m=2^20, n=2^24-2^16, a_L=a_C=0.9
    infile4 = open('eval_estimators-2021-02-03_15:04:49.txt', 'r') # m=2^20, n=2^24-2^16, a_L=a_C=0.3
    infile5 = open('eval_estimators-2021-02-03_15:12:51.txt', 'r') # m=2^20, n=2^24-2^16, a_L=a_C=0.7
    '''

    '''
    # heterogeneous multiplicative
    infile1 = open('eval_estimators-2021-02-03_16:11:24.txt', 'r') # m=2^20, n=2^24-2^16, a_L=a_C=0.5
    infile2 = open('eval_estimators-2021-02-03_16:21:10.txt', 'r') # m=2^20, n=2^24-2^16, a_L=a_C=0.1
    infile3 = open('eval_estimators-2021-02-03_16:30:24.txt', 'r') # m=2^20, n=2^24-2^16, a_L=a_C=0.9
    infile4 = open('eval_estimators-2021-02-03_16:40:05.txt', 'r') # m=2^20, n=2^24-2^16, a_L=a_C=0.3
    infile5 = open('eval_estimators-2021-02-03_16:49:06.txt', 'r') # m=2^20, n=2^24-2^16, a_L=a_C=0.7
    '''

    '''
    # heterogeneous multiplicative
    infile1 = open('eval_estimators-2021-02-07_16:56:43.txt', 'r') # m=2^22, n=2^24-2^16, a_L=a_C=0.5
    infile2 = open('eval_estimators-2021-02-07_16:58:09.txt', 'r') # m=2^20, n=2^24-2^16, a_L=a_C=0.1
    infile3 = open('eval_estimators-2021-02-07_16:58:40.txt', 'r') # m=2^20, n=2^24-2^16, a_L=a_C=0.9
    '''

    infiles = [infile1, infile2, infile3, infile4, infile5]

    # TODO: settings
    USE_NUM = True # use numerically computed values (GTE, bias, sd, etc.)
    QUIET = True   # quiet logging (GTE, bias, sd, etc.)
    USE_NUM_QUIET = False # quiet logging when using nums

    # hack for numerics: repermute the a_c / a_l sequence to match with the files
    ratios = [0.5, 0.1, 0.9, 0.3, 0.7]

    for idx, infile in enumerate(infiles):
        a_l = ratios[idx]
        a_c = ratios[idx]

        cr_bias_obs = []
        cr_nl_bias_obs = []
        lr_bias_obs = []
        lr_nl_bias_obs = []
        # tsr_bias_obs = []
        # tsris_bias_obs = {}
        cr_std_obs = []
        cr_nl_std_obs = []
        lr_std_obs = []
        lr_nl_std_obs = []

        lr_q10s = []
        lr_q11s = []
        cr_q01s = []
        cr_q11s = []

        n_steps = get_int(infile)
        for it in range(n_steps):
            n, m, phi_0, phi_1, rho = next_line(infile).split(' ')
            n = int(n)
            m = int(m)
            if not QUIET: print('=> n & m:', n, '&', m)
            phi_0 = 0.4872 # float(phi_0)# TO REMOVE
            phi_1 = 0.715 # float(phi_1) * 0.96
            rho = float(rho)

            ks = get_ks(infile)
            n_iters = get_int(infile)

            mp = Marketplace(n = n, m = m, phi_0 = phi_0, phi_1 = phi_1)
            gte = mp.GTE
            # gte = get_float(infile)
            if idx == 0:
                ns.append(n)
                ms.append(m)
                rhos.append(rho)
                if USE_NUM:
                    gtes.append(num_gte(phi_0, phi_1, rho))
                    if not USE_NUM_QUIET: print('GTE:', gtes[-1], '- sim:', gte)
                else:
                    gtes.append(gte)

            # Listing-side randomization
            # lr, lr_nl = mp.LR_bias_n_times(n_times)
            lr = get_N_floats(n_iters, infile)
            lr_nl = get_N_floats(n_iters, infile)
            lr_q10s.append(get_N_floats(n_iters, infile))
            lr_q11s.append(get_N_floats(n_iters, infile))
            if not QUIET: print('LR:', np.mean(lr) - gte, '+/-', np.std(lr))
            # print('LR NL bias:', np.mean(lr_nl) - gte, '+/-', np.std(lr_nl))
            if USE_NUM:
                lr_bias_obs.append(bias_lr(phi_0, phi_1, rho, a_l))
                if not USE_NUM_QUIET: print('LR bias num:', lr_bias_obs[-1], '- sim:', np.mean(lr) - gte)
            else:
                lr_bias_obs.append(np.mean(lr) - gte)
            lr_nl_bias_obs.append(np.mean(lr_nl) - gte)
            if USE_NUM:
                lr_std_obs.append(np.sqrt(finite_var_lr(phi_0, phi_1, rho, a_l, n)))
                if not USE_NUM_QUIET: print('LR std num:', lr_std_obs[-1], '- sim:', np.std(lr))
            else:
                lr_std_obs.append(np.std(lr))
            lr_nl_std_obs.append(np.std(lr_nl))

            # Customer-side randomization
            # cr, cr_nl = mp.CR_bias_n_times(n_times)
            cr = get_N_floats(n_iters, infile)
            cr_nl = get_N_floats(n_iters, infile)
            cr_q01s.append(get_N_floats(n_iters, infile))
            cr_q11s.append(get_N_floats(n_iters, infile))
            if not QUIET: print('CR:', np.mean(cr) - gte, '+/-', np.std(cr))
            # print('CR NL:', np.mean(cr_nl) - gte, '+/-', np.std(cr_nl))
            if USE_NUM:
                cr_bias_obs.append(bias_cr(phi_0, phi_1, rho, a_c))
                if not USE_NUM_QUIET: print('CR bias num:', cr_bias_obs[-1], '- sim:', np.mean(cr) - gte)
            else:
                cr_bias_obs.append(np.mean(cr) - gte)
            cr_nl_bias_obs.append(np.mean(cr_nl) - gte)
            if USE_NUM:
                cr_std_obs.append(np.sqrt(finite_var_cr(phi_0, phi_1, rho, a_c, n)))
                if not USE_NUM_QUIET: print('CR std num:', cr_std_obs[-1], '- sim:', np.std(cr))
            else:
                cr_std_obs.append(np.std(cr))
            cr_nl_std_obs.append(np.std(cr_nl))

            # Two-sided randomization
            # tsr, tsr_imp_1, tsr_imp_2 = mp.TSR_bias_n_times(n_times, a_c=a_c, a_l=a_l, beta=beta, ks=[1,2])
            '''
            tsr = get_N_floats(n_iters, infile)
            tsris = {k : get_N_floats(n_iters, infile) for k in ks}
            tsr_q00 = get_N_floats(n_iters, infile)
            tsr_q01 = get_N_floats(n_iters, infile)
            tsr_q10 = get_N_floats(n_iters, infile)
            tsr_q11 = get_N_floats(n_iters, infile)
            # beta = np.exp(-rho)
            # a_c = 1 - beta + 0.5 * beta
            # a_l = 0.5 * (1 - beta) + beta
            # tsr = np.array(tsr_q11) / a_c / a_l / n - (np.array(tsr_q01) + np.array(tsr_q10) + np.array(tsr_q00)) / (1 - a_c * a_l) / n
            print('TSR:', np.mean(tsr) - gte, '+/-', np.std(tsr))
            tsr_bias_obs.append(np.mean(tsr) - gte)
            tsr_std_obs.append(np.std(tsr))
            for k, tsri in tsris.items():
                print('TSRI-' + str(k) + ':', np.mean(tsri) - gte, '+/-', np.std(tsri))
                if k not in tsris_bias_obs: tsris_bias_obs[k] = []
                if k not in tsris_std_obs: tsris_std_obs[k] = []
                tsris_bias_obs[k].append(np.mean(tsri) - gte)
                tsris_std_obs[k].append(np.std(tsri))
            '''

        infile.close()
        lr_bias_obs_series.append(lr_bias_obs)
        lr_std_obs_series.append(lr_std_obs)
        cr_bias_obs_series.append(cr_bias_obs)
        cr_std_obs_series.append(cr_std_obs)
        lr_q10_m.append(lr_q10s)
        lr_q11_m.append(lr_q11s)
        cr_q01_m.append(cr_q01s)
        cr_q11_m.append(cr_q11s)

    # legend_labels = ['LR', 'LRNL', 'CR', 'CRNL', 'TSR'] + ['TSRI-' + str(k) for k in ks]
    legend_labels = ['LR', 'CR']
    # bias_series = [lr_bias_obs, lr_nl_bias_obs, cr_bias_obs, cr_nl_bias_obs, tsr_bias_obs] + tsris_bias_obs.values()
    bias_series = [lr_bias_obs_series, cr_bias_obs_series]
    # std_series = [lr_std_obs, lr_nl_std_obs, cr_std_obs, cr_nl_std_obs, tsr_std_obs] + tsris_std_obs.values()
    std_series = [lr_std_obs_series, cr_std_obs_series]
    bias_series = [np.array(serie) for serie in bias_series]
    std_series = [np.array(serie) for serie in std_series]
    mse_series = [np.sqrt(b ** 2 + s ** 2) for b,s in zip(bias_series, std_series)]

    '''
    # Relative bias
    plt.subplot(3, 2, 1)
    series = []
    for serie in bias_series:
        series.append(plt.plot(rhos, np.abs(serie) / np.array(gtes))[0])
        # series.append(plt.plot(ns, np.abs(serie) / np.array(gtes))[0])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('|bias/GTE|')
    plt.legend(series, legend_labels)
    '''

    '''
    # Normalized bias
    plt.subplot(1, 3, 1)
    series = []
    for idx, serie in enumerate(bias_series):
        series.append(plt.plot(rhos[2:-2], np.abs(serie[0, 2:-2]) / np.array(gtes[2:-2]), color=colorWheel[idx])[0])
        plt.fill_between(rhos[2:-2], np.abs(np.min(serie[:, 2:-2], axis=0)) / np.array(gtes[2:-2]), np.abs(np.max(serie[:, 2:-2], axis=0)) / np.array(gtes[2:-2]), color=colorWheel[idx], alpha=0.2)
        # plt.plot(rhos, np.abs(np.max(serie, axis=0)) / np.minimum(rhos, 1.0) / np.array(gtes), ls=':', color=colorWheel[idx])
        # plt.plot(rhos, np.abs(np.min(serie, axis=0)) / np.minimum(rhos, 1.0) / np.array(gtes), ls=':', color=colorWheel[idx])
        # series.append(plt.plot(ns, np.abs(serie) / np.minimum(1.0, rhos))[0])

    # series.append(plt.plot(rhos[:], np.array(gtes)[:] / np.minimum(rhos[:], 1.0), color=colorWheel[len(bias_series)], ls=':')[0])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Relative demand $\lambda$')
    plt.ylabel('Bias/GTE')
    plt.legend(series, ['LR', 'CR']) # , r'$GTE / \min\{\lambda, 1\}$'])
    '''

    '''
    # Relative scaled sqrt variance
    plt.subplot(3, 2, 3)
    series = []
    for serie in std_series:
        series.append(plt.plot(rhos, np.array(serie) / np.array(gtes))[0])
        # series.append(plt.plot(ns, np.sqrt(ns) * np.array(serie) / np.array(gtes))[0])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('sqrt(n) Stdev/GTE')
    plt.legend(series, legend_labels)
    '''

    # Normalized scaled sqrt variance
    # plt.subplot(1, 3, 2)
    series = []
    for idx, serie in enumerate(std_series):
        series.append(plt.plot(rhos[2:-2], np.array(serie[0, 2:-2]) / np.array(gtes[2:-2]), color=colorWheel[idx])[0])
        plt.fill_between(rhos[2:-2], np.min(serie[:, 2:-2], axis=0) / np.array(gtes[2:-2]), np.max(serie[:, 2:-2], axis=0) / np.array(gtes[2:-2]), color=colorWheel[idx], alpha=0.2)
        # plt.plot(rhos, np.max(serie, axis=0) / np.minimum(rhos, 1.0) / np.array(gtes), ls=':', color=colorWheel[idx])
        # plt.plot(rhos, np.min(serie, axis=0) / np.minimum(rhos, 1.0) / np.array(gtes), ls=':', color=colorWheel[idx])
        # series.append(plt.plot(ns, np.sqrt(ns) * np.array(serie) / np.minimum(1.0, rhos))[0])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Relative demand $\lambda$')
    plt.ylabel('SD/GTE')
    plt.legend(series, legend_labels)

    '''
    # Relative MSE
    plt.subplot(3, 2, 5)
    series = []
    for serie in mse_series:
        series.append(plt.plot(rhos, np.abs(serie) / np.array(gtes))[0])
        # series.append(plt.plot(ns, np.abs(serie) / np.array(gtes))[0])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('MSE/GTE')
    plt.legend(series, legend_labels)
    '''

    '''
    # Normalized MSE
    plt.subplot(1, 3, 3)
    series = []
    for idx, serie in enumerate(mse_series):
        series.append(plt.plot(rhos[2:-2], np.abs(serie[0, 2:-2]) / np.array(gtes[2:-2]), color=colorWheel[idx])[0])
        plt.fill_between(rhos[2:-2], np.min(serie[:, 2:-2], axis=0) / np.array(gtes[2:-2]), np.max(serie[:, 2:-2], axis=0) / np.array(gtes[2:-2]), color=colorWheel[idx], alpha=0.2)
        # plt.plot(rhos, np.max(serie, axis=0) / np.minimum(rhos, 1.0) / np.array(gtes), ls=':', color=colorWheel[idx])
        # plt.plot(rhos, np.min(serie, axis=0) / np.minimum(rhos, 1.0) / np.array(gtes), ls=':', color=colorWheel[idx])
        # series.append(plt.plot(ns, np.abs(serie) / np.minimum(1.0, rhos))[0])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Relative demand $\lambda$')
    plt.ylabel('RMSE/GTE')
    plt.legend(series, legend_labels)
    '''

    sns.despine()
    plt.tight_layout()
    plt.show()

