#!/usr/bin/env python
'''
Plot evolution of bias and SD (normalized by GTE) as market size increases,
while fixing the same balance and consideration rate.
'''

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

    #                                                          rho - steps x iters max_n rho_0-rho_1
    infile1 = open('bv_tradeoff-2021-02-10_01:20:23.txt', 'r') # 1:1 - 14x1000 16M- 20-22
    infile2 = open('bv_tradeoff-2021-02-10_01:16:54.txt', 'r') # 1:8 - 14x1000 16M- 20-22
    infile3 = open('bv_tradeoff-2021-02-10_01:03:49.txt', 'r') # 8:1 - 14x1000 16M- 20-22

    '''
    infile2 = open('bv_tradeoff-2021-02-11_00:30:48.txt', 'r') # 16:1 - 10x1000
    infile1 = open('bv_tradeoff-2021-02-11_10:52:01.txt', 'r') # 1:1 - 10x1000
    # infile3 = open('bv_tradeoff-2021-02-11_00:37:33.txt', 'r') # 1:16 - 10x1000
    infile3 = open('bv_tradeoff-2021-02-11_12:01:53.txt', 'r') # 1:16 - 10x1000
    '''

    infiles = [infile2, infile1, infile3]

    xlabels = [r'$N = 8M$', r'$N = M$', r'$N = M/8$']

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

        ns = []
        ms = []
        rhos = []
        gtes = []

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

            # mp = Marketplace(n = n, m = m, phi_0 = phi_0, phi_1 = phi_1)
            # gte = mp.GTE
            gte = get_float(infile)
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
        bias_series = [cr_bias_obs_series, lr_bias_obs_series]
        # std_series = [lr_std_obs, lr_nl_std_obs, cr_std_obs, cr_nl_std_obs, tsr_std_obs] + tsris_std_obs.values()
        std_series = [lr_std_obs_series, cr_std_obs_series]
        bias_series = [np.array(serie) for serie in bias_series]
        std_series = [np.array(serie) for serie in std_series]
        mse_series = [np.sqrt(b ** 2 + s ** 2) for b,s in zip(bias_series, std_series)]

        # plt.subplot(1, 3, idx+1)
        if idx == 1:
            plt.plot(ns, np.abs(bias_series[0][idx]), color=colorWheel[0])
            plt.plot(ns, np.abs(bias_series[1][idx]), color=colorWheel[1])
            plt.plot(ns, std_series[0][idx], color=colorWheel[0], ls=':')
            plt.plot(ns, std_series[1][idx], color=colorWheel[1], ls=':')

            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$N$')
            # plt.xlabel(xlabels[idx])
            # plt.ylabel(r'Relative bias')
            plt.legend(['CR Bias/GTE', 'LR Bias/GTE', 'CR SD/GTE', 'LR SD/GTE']) # , r'$GTE / \min\{\lambda, 1\}$'])

    sns.despine()
    plt.show()

