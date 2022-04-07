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

    # infile1 = open('eval_estimators-2021-02-03_00:22:46.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.5
    # infile2 = open('eval_estimators-2021-02-03_00:23:53.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.1
    # infile3 = open('eval_estimators-2021-02-03_00:25:51.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.9
    # infile4 = open('eval_estimators-2021-02-03_00:27:22.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.3
    # infile5 = open('eval_estimators-2021-02-03_00:28:28.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.7

    '''
    infile1 = open('eval_estimators-2021-02-03_10:29:02.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.5
    infile2 = open('eval_estimators-2021-02-03_12:54:19.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.1
    infile3 = open('eval_estimators-2021-02-03_12:55:21.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.9
    infile4 = open('eval_estimators-2021-02-03_12:56:15.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.3
    infile5 = open('eval_estimators-2021-02-03_12:56:56.txt', 'r') # m=2^22, n=2^28-2^16, a_L=a_C=0.7
    '''

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

    infile1 = open('eval_estimators-2021-02-07_16:43:01.txt', 'r') # m=n=2^23, a_L=a_C=0.1, vary treatment
    infile2 = open('eval_estimators-2021-02-07_16:41:49.txt', 'r') # m=n=2^23, a_L=a_C=0.5, vary treatment
    infile3 = open('eval_estimators-2021-02-07_16:46:09.txt', 'r') # m=n=2^23, a_L=a_C=0.9, vary treatment

    infile1 = open('eval_estimators-2021-02-09_16:30:01.txt', 'r') # m=n=2^24, a_L=a_C=0.1, vary treatment
    infile2 = open('eval_estimators-2021-02-09_16:30:32.txt', 'r') # m=n=2^24, a_L=a_C=0.5, vary treatment
    infile3 = open('eval_estimators-2021-02-09_16:32:58.txt', 'r') # m=n=2^24, a_L=a_C=0.9, vary treatment

    infile1 = open('treatment_vary-2021-02-12_00:58:16.txt', 'r') # m=n=2^20, a_L=a_C=0.1
    infile2 = open('treatment_vary-2021-02-12_00:54:11.txt', 'r') # m=n=2^20, a_L=a_C=0.5
    infile3 = open('treatment_vary-2021-02-12_01:00:41.txt', 'r') # m=n=2^20, a_L=a_C=0.9

    infile1 = open('treatment_vary-2021-02-12_01:14:42.txt', 'r') # m=4n=2^20, a_L=a_C=0.1
    infile2 = open('treatment_vary-2021-02-12_01:12:55.txt', 'r') # m=4n=2^20, a_L=a_C=0.5
    infile3 = open('treatment_vary-2021-02-12_01:15:43.txt', 'r') # m=4n=2^20, a_L=a_C=0.9

    infile1 = open('treatment_vary-2021-02-12_01:18:03.txt', 'r') # m=n/4=2^20, a_L=a_C=0.1
    infile2 = open('treatment_vary-2021-02-12_01:17:41.txt', 'r') # m=n/4=2^20, a_L=a_C=0.5
    infile3 = open('treatment_vary-2021-02-12_01:21:37.txt', 'r') # m=n/4=2^20, a_L=a_C=0.9

    infile1 = open('treatment_vary-2021-02-12_01:33:21.txt', 'r') # m=n=2^20, a_L=a_C=0.1, het L
    infile2 = open('treatment_vary-2021-02-12_01:34:02.txt', 'r') # m=n=2^20, a_L=a_C=0.5, het L
    infile3 = open('treatment_vary-2021-02-12_01:34:14.txt', 'r') # m=n=2^20, a_L=a_C=0.9, het L

    infile1 = open('treatment_vary-2021-02-12_01:49:07.txt', 'r') # m=n=2^20, a_L=a_C=0.1, het C
    infile2 = open('treatment_vary-2021-02-12_01:48:37.txt', 'r') # m=n=2^20, a_L=a_C=0.5, het C
    infile3 = open('treatment_vary-2021-02-12_01:49:21.txt', 'r') # m=n=2^20, a_L=a_C=0.9, het C

    infiles = [infile1, infile2, infile3]

    for idx, infile in enumerate(infiles):
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
            print('=> n & m:', n, '&', m)
            phi_0 = float(phi_0)
            phi_1 = float(phi_1)
            rho = float(rho)

            ks = get_ks(infile)
            n_iters = get_int(infile)

            gte = get_float(infile)
            # mp = Marketplace(n = n, m = m, phi_0 = phi_0, phi_1 = phi_1)
            # gte = mp.GTE
            if idx == 0:
                ns.append(n)
                ms.append(m)
                rhos.append(rho)
                gtes.append(gte)

            # Listing-side randomization
            # lr, lr_nl = mp.LR_bias_n_times(n_times)
            lr = get_N_floats(n_iters, infile)
            lr_nl = get_N_floats(n_iters, infile)
            lr_q10s.append(get_N_floats(n_iters, infile))
            lr_q11s.append(get_N_floats(n_iters, infile))
            print('LR:', np.mean(lr) - gte, '+/-', np.std(lr))
            # print('LR NL bias:', np.mean(lr_nl) - gte, '+/-', np.std(lr_nl))
            lr_bias_obs.append(np.mean(lr) - gte)
            lr_nl_bias_obs.append(np.mean(lr_nl) - gte)
            lr_std_obs.append(np.std(lr))
            lr_nl_std_obs.append(np.std(lr_nl))

            # Customer-side randomization
            # cr, cr_nl = mp.CR_bias_n_times(n_times)
            cr = get_N_floats(n_iters, infile)
            cr_nl = get_N_floats(n_iters, infile)
            cr_q01s.append(get_N_floats(n_iters, infile))
            cr_q11s.append(get_N_floats(n_iters, infile))
            print('CR:', np.mean(cr) - gte, '+/-', np.std(cr))
            # print('CR NL:', np.mean(cr_nl) - gte, '+/-', np.std(cr_nl))
            cr_bias_obs.append(np.mean(cr) - gte)
            cr_nl_bias_obs.append(np.mean(cr_nl) - gte)
            cr_std_obs.append(np.std(cr))
            cr_nl_std_obs.append(np.std(cr_nl))

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
    bias_series = [cr_bias_obs_series, lr_bias_obs_series]
    # std_series = [lr_std_obs, lr_nl_std_obs, cr_std_obs, cr_nl_std_obs, tsr_std_obs] + tsris_std_obs.values()
    std_series = [cr_std_obs_series, lr_std_obs_series]
    bias_series = [np.array(serie) for serie in bias_series]
    std_series = [np.array(serie) for serie in std_series]
    mse_series = [np.sqrt(b ** 2 + s ** 2) for b,s in zip(bias_series, std_series)]

    rhos = np.array(rhos)
    gtes = np.array(gtes)

    # treatment_effects = [1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5] # omit 1.0
    treatment_effects = [1.1, 1.2, 1.3, 1.4, 1.5] # omit 1.0

    plt.rc('figure', figsize=[9.6, 4.0])

    plt.subplot(1, 3, 1)
    series = []
    for idx, serie in enumerate(bias_series):
        series.append(plt.plot(treatment_effects[:5], serie[1, :5] / gtes[:5], color=colorWheel[idx])[0])
        plt.fill_between(treatment_effects[:5], np.min(serie[:, :5], axis=0) / np.array(gtes[:5]), np.max(serie[:, :5], axis=0) / np.array(gtes[:5]), alpha=0.1, color=colorWheel[idx])

    plt.xlabel(r'Treatment coefficient $\alpha$')
    plt.ylabel('Bias/GTE')
    plt.legend(series, ['CR', 'LR'])

    plt.subplot(1, 3, 2)
    series = []
    for idx, serie in enumerate(std_series):
        series.append(plt.plot(treatment_effects[:5], serie[1, :5] / gtes[:5], color=colorWheel[idx])[0])
        plt.fill_between(treatment_effects[:5], np.min(serie[:, :5], axis=0) / np.array(gtes[:5]), np.max(serie[:, :5], axis=0) / np.array(gtes[:5]), alpha=0.1, color=colorWheel[idx])

    plt.xlabel(r'Treatment coefficient $\alpha$')
    plt.ylabel('SD/GTE')
    plt.legend(series, ['CR', 'LR'])

    plt.subplot(1, 3, 3)
    series = []
    for idx, serie in enumerate(mse_series):
        series.append(plt.plot(treatment_effects[:5], serie[1, :5] / gtes[:5], color=colorWheel[idx])[0])
        plt.fill_between(treatment_effects[:5], np.min(serie[:, :5], axis=0) / np.array(gtes[:5]), np.max(serie[:, :5], axis=0) / np.array(gtes[:5]), alpha=0.1, color=colorWheel[idx])

    plt.xlabel(r'Treatment coefficient $\alpha$')
    plt.ylabel('MSE/GTE')
    plt.legend(series, ['CR', 'LR'])

    '''
    for idx, serie in enumerate(bias_series):
        plt.subplot(1, 2, idx + 1)
        series = []
        for i, s in enumerate(serie):
            series.append(plt.plot(treatment_effects[-5:], serie[i, -5:] / gtes[-5:], color=colorWheel[i])[0])
            plt.fill_between(treatment_effects[-5:], (serie[i, -5:] - 2 * std_series[idx][i, -5:]) / gtes[-5:], (serie[i] + 2 * std_series[idx][i] * np.sign(gtes)) / gtes, alpha = 0.2, color=colorWheel[i])
            # series.append(plt.plot(ns, np.abs(serie) / np.minimum(1.0, rhos))[0])

        # series.append(plt.plot(treatment_effects, gtes, color=colorWheel[-1])[0])
        # plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel(r'Treatment coefficient $\alpha$')
        plt.ylabel(legend_labels[idx] + ' relative bias')
        plt.legend(series, [r'$10\%$ treatment', r'$50\%$ treatment', r'$90\%$ treatment', 'GTE'])
    '''

    sns.despine()
    plt.show()

    '''
    plt.plot(treatment_effects, (np.max(bias_series[1], axis=0) - np.min(bias_series[1], axis=0)) / np.abs(gtes), marker='.', color=colorWheel[0])
    plt.plot(treatment_effects, ((np.array(treatment_effects) - 1) * phi_0) ** 2 / np.abs(gtes), ls=':', color=colorWheel[-1])
    plt.xlabel(r'Treatment coefficient $\alpha$')
    plt.legend(['CR (max bias - min bias) / GTE', 'Bound on CR relative bias differential'])
    plt.show()
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

    # Normalized scaled sqrt variance
    plt.subplot(1, 3, 2)
    series = []
    for idx, serie in enumerate(std_series):
        series.append(plt.plot(treatment_effects, np.array(serie[0]) / np.minimum(rhos, 1.0) / gtes, color=colorWheel[idx])[0])
        plt.plot(treatment_effects, np.max(serie[1:], axis=0) / np.minimum(rhos, 1.0) / gtes, ls=':', color=colorWheel[idx])
        plt.plot(treatment_effects, np.min(serie[1:], axis=0) / np.minimum(rhos, 1.0) / gtes, ls=':', color=colorWheel[idx])
        # series.append(plt.plot(ns, np.sqrt(ns) * np.array(serie) / np.minimum(1.0, rhos))[0])

    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r'Treatment coefficient $\alpha$')
    plt.ylabel(r'$\sqrt{N}$ stdev')
    plt.legend(series, legend_labels)

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

    # Normalized MSE
    plt.subplot(1, 3, 3)
    series = []
    for idx, serie in enumerate(mse_series):
        series.append(plt.plot(treatment_effects, np.abs(serie[0]) / np.minimum(rhos, 1.0) / gtes, color=colorWheel[idx])[0])
        plt.plot(treatment_effects, np.max(serie[1:], axis=0) / np.minimum(rhos, 1.0) / gtes, ls=':', color=colorWheel[idx])
        plt.plot(treatment_effects, np.min(serie[1:], axis=0) / np.minimum(rhos, 1.0) / gtes, ls=':', color=colorWheel[idx])
        # series.append(plt.plot(ns, np.abs(serie) / np.minimum(1.0, rhos))[0])

    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r'Treatment coefficient $\alpha$')
    plt.ylabel('MSE')
    plt.legend(series, legend_labels)
    '''


