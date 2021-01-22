#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from Marketplace import Marketplace

def get_int(infile):
    return int(next(infile))

def get_float(infile):
    return float(next(infile))

def get_ks(infile):
    return next(infile).strip().split(' ')[1:]

def get_N_ints(n, infile):
    return [int(next(infile)) for _ in range(n)]

def get_N_floats(n, infile):
    return [float(next(infile)) for _ in range(n)]

if __name__ == '__main__':
    # m = 1093500 # 2^16
    # n = 64000 # 2^11
    # n_times = 1000

    cr_bias_obs = []
    cr_nl_bias_obs = []
    lr_bias_obs = []
    lr_nl_bias_obs = []
    tsr_bias_obs = []
    tsris_bias_obs = {}
    cr_std_obs = []
    cr_nl_std_obs = []
    lr_std_obs = []
    lr_nl_std_obs = []
    tsr_std_obs = []
    tsris_std_obs = {}

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
    infile = open('eval_estimators-2020-10-22_17:42:57.txt', 'r') # 1:1 - 14x20000 16M- 10-13pct
    # infile = open('eval_estimators-2020-09-26_11:36:14.txt', 'r') # 1:1 - 14x10000
    # infile = open('eval_estimators-2020-09-27_22:04:45.txt', 'r') # 1:1 - 14x10000 large size
    # infile = open('eval_estimators-2020-09-26_16:30:50.txt', 'r') # 8:1 - 14x10000
    # infile = open('eval_estimators-2020-09-27_22:20:40.txt', 'r') # 8:1 - 14x10000 large size
    # infile = open('eval_estimators-2020-09-26_16:54:16.txt', 'r') # 1:8 - 14x10000
    # infile = open('eval_estimators-2020-10-02_15:19:06.txt', 'r') # 1:1 - 4x50000 small market
    n_steps = get_int(infile)
    for it in range(n_steps):
        n, m, phi_0, phi_1, rho = next(infile).strip().split(' ')
        n = int(n)
        m = int(m)
        print('=> n & m:', n, '&', m)
        phi_0 = float(phi_0)
        phi_1 = float(phi_1)
        rho = float(rho)
        mp = Marketplace(n = n, m = m, phi_0 = phi_0, phi_1 = phi_1)
        gte = mp.GTE
        ns.append(n)
        ms.append(m)
        rhos.append(rho)
        gtes.append(gte)

        ks = get_ks(infile)
        n_iters = get_int(infile)

        # Listing-side randomization
        # lr, lr_nl = mp.LR_bias_n_times(n_times)
        lr = get_N_floats(n_iters, infile)
        lr_nl = get_N_floats(n_iters, infile)
        lr_q10 = get_N_floats(n_iters, infile)
        lr_q11 = get_N_floats(n_iters, infile)
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
        cr_q01 = get_N_floats(n_iters, infile)
        cr_q11 = get_N_floats(n_iters, infile)
        print('CR:', np.mean(cr) - gte, '+/-', np.std(cr))
        # print('CR NL:', np.mean(cr_nl) - gte, '+/-', np.std(cr_nl))
        cr_bias_obs.append(np.mean(cr) - gte)
        cr_nl_bias_obs.append(np.mean(cr_nl) - gte)
        cr_std_obs.append(np.std(cr))
        cr_nl_std_obs.append(np.std(cr_nl))

        # Two-sided randomization
        # tsr, tsr_imp_1, tsr_imp_2 = mp.TSR_bias_n_times(n_times, a_c=a_c, a_l=a_l, beta=beta, ks=[1,2])
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

        # m = int(m / 3 * 2)
        # n = int(n / 2 * 3)

    infile.close()

    # legend_labels = ['LR', 'LRNL', 'CR', 'CRNL', 'TSR'] + ['TSRI-' + str(k) for k in ks]
    legend_labels = ['LR', 'CR', 'TSR']
    # bias_series = [lr_bias_obs, lr_nl_bias_obs, cr_bias_obs, cr_nl_bias_obs, tsr_bias_obs] + tsris_bias_obs.values()
    bias_series = [lr_bias_obs, cr_bias_obs, tsr_bias_obs]
    # std_series = [lr_std_obs, lr_nl_std_obs, cr_std_obs, cr_nl_std_obs, tsr_std_obs] + tsris_std_obs.values()
    std_series = [lr_std_obs, cr_std_obs, tsr_std_obs]
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
    mp = Marketplace(n=4096, m=16384, phi_0=0.24, phi_1=0.3)
    cr, cr_nl = mp.CR_bias_n_times(500)
    print('CR:', np.mean(cr), '+/-', np.std(cr))
    print('CR NL:', np.mean(cr_nl), '+/-', np.std(cr_nl))
    lr, lr_nl = mp.LR_bias_n_times(500)
    print('LR:', np.mean(lr), '+/-', np.std(lr))
    print('LR NL:', np.mean(lr_nl), '+/-', np.std(lr_nl))
    tsr = mp.TSR_bias_n_times(500)
    print('TSR:', np.mean(tsr), '+/-', np.std(tsr))
    plt.boxplot([cr, cr_nl, lr, lr_nl, tsr])
    plt.show()
    '''

