#!/usr/bin/env python
'''
Analyze lrcr_vary_design-* output files.
'''

import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from Marketplace import Marketplace
from common import *

plt.rcParams["savefig.dpi"] = 200

if __name__ == '__main__':

    # infile = open('lrcr_vary_design-2020-11-05_13:26:48.txt', 'r') # 0.5M, 20k avg, 0.01 GTE, full
    # infile = open('lrcr_vary_design-2020-11-15_23:12:31.txt', 'r') # 0.5M, 20k avg, 0.01 GTE, full, rho=9.02
    # infile = open('lrcr_vary_design-2020-11-16_09:49:45.txt', 'r') # 0.5M/2, 20k avg, 0.01 GTE, full, rho=18.04
    # infile = open('lrcr_vary_design-2020-11-16_14:07:54.txt', 'r') # 0.5M, 50k avg, 0.01 GTE, full, rho=18.04
    # infile = open('lrcr_vary_design-2020-12-03_14:20:21.txt', 'r') # 0.5M, 50k avg, 0.01 GTE, full, rho=18.04, poisson approx
    # infile = open('lrcr_vary_design-2020-12-03_14:20:21.txt', 'r') # 0.5M, 50k avg, 0.01 GTE, full, rho=18.04, poisson approx
    # infile = open('lrcr_vary_design-2020-12-07_11:16:58.txt', 'r') # 0.5M, 50k avg, 0.1 GTE, full, rho=18.04, poisson approx
    # infile = open('lrcr_vary_design-2020-12-07_20:19:26.txt', 'r') # 0.5M, 50k avg, 0.1 GTE, full, rho=9.02, poisson approx, LR
    # infile = open('lrcr_vary_design-2020-12-10_10:08:54.txt', 'r') # 0.5M, 50k avg, 0.1 GTE, full, rho=6.35, poisson approx, LR
    # infile = open('lrcr_vary_design-2021-01-03_16:32:15.txt', 'r') # 0.5M, 50k avg, 0.05 GTE, full, rho=2, LR
    # infile = open('lrcr_vary_design-2021-01-03_16:30:02.txt', 'r') # 0.5M, 50k avg, 0.05 GTE, full, rho=10, LR

    # infile = open('lrcr_vary_design-2021-01-21_20:16:53.txt', 'r') # 1M, 10k avg, 0.05 GTE, full, rho=1, LRCR (.1, .5, .9)
    # infile = open('lrcr_vary_design-2021-01-24_17:24:38.txt', 'r') # 0.1M, 10k avg, 0.05 GTE, full, rho=1, LRCR (.1, .5, .9)
    # infile = open('lrcr_vary_design-2021-01-24_17:20:48.txt', 'r') # 1M, 10k avg, 0.02 GTE, full, rho=1, LRCR (.1, .5, .9)
    # infile = open('lrcr_vary_design-2021-02-08_17:09:33.txt', 'r') # 10M, 10k avg, 0.02 GTE, full, rho=1, LRCR (.1, .3, .5, .7, .9)
    infile1 = open('lrcr_vary_design-2021-02-09_10:57:34.txt', 'r') # 100k, 10k avg, 0.02 GTE, full, rho=1, LRCR (.1, ..., .9)
    infile2 = open('lrcr_vary_design-2021-02-09_10:56:03.txt', 'r') # 1M, 10k avg, 0.02 GTE, full, rho=1, LRCR (.1, ..., .9)
    infile3 = open('lrcr_vary_design-2021-02-09_11:00:16.txt', 'r') # 10M, 1k avg, 0.02 GTE, full, rho=1, LRCR (.1, ..., .9)

    for infile, k_samples in zip([infile1, infile2, infile3], [9000, 3000, 1000]):

        cr_bias_obs = []
        cr_nl_bias_obs = []
        lr_bias_obs = []
        lr_nl_bias_obs = []
        cr_std_obs = []
        cr_nl_std_obs = []
        lr_std_obs = []
        lr_nl_std_obs = []

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
                # lr_bias_obs.append(np.mean(lr) - gte)
                lr_nl_bias_obs.append(np.mean(lr_nl) - gte)
                lr_std_obs.append(np.std(lr[:k_samples]))
                # lr_std_obs.append(np.std(lr))
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
                # cr_bias_obs.append(np.mean(cr) - gte)
                cr_nl_bias_obs.append(np.mean(cr_nl) - gte)
                cr_std_obs.append(np.std(cr[:k_samples]))
                # cr_std_obs.append(np.std(cr))
                cr_nl_std_obs.append(np.std(cr_nl))

        infile.close()

        a_ls = np.array(a_ls)
        a_cs = np.array(a_cs)
        n_a_ls = int(np.round(a_ls.max() / a_ls.min()))
        n_a_cs = int(np.round(a_cs.max() / a_cs.min()))

        plt.rc('figure', figsize=[9.6, 4.0])

        plt.subplot(1,3,1)
        plt.plot(a_cs[:(len(a_cs)//2)], np.array(cr_bias_obs) / gte, color=colorWheel[0], marker='.')
        plt.plot(a_cs[:(len(a_cs)//2)], np.array(lr_bias_obs) / gte, color=colorWheel[1], marker='.')
        plt.xlabel(r'Treatment allocation')
        plt.ylabel('Bias/GTE')
        plt.legend(['CR', 'LR'])

        plt.subplot(1,3,2)
        plt.plot(a_cs[:(len(a_cs)//2)], np.array(cr_std_obs) / gte, color=colorWheel[0], marker='.')
        plt.plot(a_cs[:(len(a_cs)//2)], np.array(lr_std_obs) / gte, color=colorWheel[1], marker='.')
        plt.xlabel(r'Treatment allocation')
        plt.ylabel('SD/GTE')
        plt.legend(['CR', 'LR'])

        plt.subplot(1,3,3)
        plt.plot(a_cs[:(len(a_cs)//2)], np.sqrt(np.array(cr_std_obs) ** 2 + np.array(cr_bias_obs) ** 2) / gte, color=colorWheel[0], marker='.')
        plt.plot(a_cs[:(len(a_cs)//2)], np.sqrt(np.array(lr_std_obs) ** 2 + np.array(lr_bias_obs) ** 2) / gte, color=colorWheel[1], marker='.')
        plt.xlabel(r'Treatment allocation')
        plt.ylabel('RMSE/GTE')
        plt.legend(['CR', 'LR'])

        sns.despine()
        plt.tight_layout()
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

