import numpy as np
import matplotlib.pyplot as plt
from Marketplace import Marketplace
import seaborn as sns
import os

def G(x): return 1-np.exp(-x)

def F(x): return G(x) / x

def num_gte(conrate, tildeconrate, lmbda):
    return (G(lmbda * G(tildeconrate)) - G(lmbda * G(conrate))) / lmbda

def bias_lr(conrate, tildeconrate, lmbda, a_l):
    ngte = num_gte(conrate, tildeconrate, lmbda)
    barconrate = (1-a_l) * conrate + a_l * tildeconrate
    return (G(lmbda * tildeconrate * F(barconrate)) - G(lmbda * conrate * F(barconrate))) / lmbda - ngte

def bias_cr(conrate, tildeconrate, lmbda, a_c):
    ngte = num_gte(conrate, tildeconrate, lmbda)
    apprate = G(conrate)
    tildeapprate = G(tildeconrate)
    barapprate = (1-a_c) * apprate + a_c * tildeapprate
    return (tildeapprate - apprate) * F(lmbda * barapprate) - ngte

def vt_lr(conrate, tildeconrate, lmbda, a_l):
    g = G(lmbda * tildeconrate * F((1-a_l) * conrate + a_l * tildeconrate))
    return g * (1-g) / a_l

def vc_lr(conrate, tildeconrate, lmbda, a_l):
    g = G(lmbda * conrate * F((1-a_l) * conrate + a_l * tildeconrate))
    return g * (1-g) / (1-a_l)

def cv_lr(conrate, tildeconrate, lmbda, a_l):
    return -lmbda * (
                tildeconrate * (1 - G(lmbda * tildeconrate * F((1-a_l) * conrate + a_l * tildeconrate))) -
                conrate * (1 - G(lmbda * conrate * F((1-a_l) * conrate + a_l * tildeconrate)))
            ) ** 2

def nvar_lr(conrate, tildeconrate, lmbda, a_l):
    return vt_lr(conrate, tildeconrate, lmbda, a_l) + \
        vc_lr(conrate, tildeconrate, lmbda, a_l) + \
        cv_lr(conrate, tildeconrate, lmbda, a_l)

def vt_cr(apprate, tildeapprate, barapprate, lmbda, a_c):
    g = tildeapprate / barapprate * G(lmbda * barapprate)
    return g * (1 - a_c * g) / a_c

def vc_cr(apprate, tildeapprate, barapprate, lmbda, a_c):
    g = apprate / barapprate * G(lmbda * barapprate)
    return g * (1 - (1-a_c) * g) / (1-a_c)

def cvtt_cr(apprate, tildeapprate, barapprate, lmbda, a_c):
    t1 = -(1-a_c) * apprate * tildeapprate * F(lmbda * barapprate)**2 * lmbda / barapprate
    t2 = -a_c * tildeapprate**2 * lmbda * np.exp(-2 * lmbda * barapprate) / barapprate
    t3 = (1-a_c) * a_c * lmbda * apprate * (1 - apprate) * \
            (-tildeapprate * F(lmbda * barapprate) / barapprate + tildeapprate * np.exp(-lmbda * barapprate) / barapprate)**2
    t4 = lmbda * tildeapprate * (1 - tildeapprate) * \
            ((1-a_c) * apprate * F(lmbda * barapprate) / barapprate + a_c * tildeapprate * np.exp(-lmbda * barapprate) / barapprate)**2
    return (t1 + t2 + t3 + t4) / a_c

def cvcc_cr(apprate, tildeapprate, barapprate, lmbda, a_c):
    t1 = -a_c * apprate * tildeapprate * F(lmbda * barapprate)**2 * lmbda / barapprate
    t2 = -(1-a_c) * apprate**2 * lmbda * np.exp(-2 * lmbda * barapprate) / barapprate
    t3 = lmbda * apprate * (1 - apprate) * \
            (a_c * tildeapprate * F(lmbda * barapprate) / barapprate + (1-a_c) * apprate * np.exp(-lmbda * barapprate) / barapprate)**2
    t4 = a_c * (1-a_c) * lmbda * tildeapprate * (1 - apprate) * \
            (-apprate * F(lmbda * barapprate) / barapprate + apprate * np.exp(-lmbda * barapprate) / barapprate)**2
    return (t1 + t2 + t3 + t4) / (1-a_c)

def cvtc_cr(apprate, tildeapprate, barapprate, lmbda, a_c):
    cvself_cr = 2 * apprate * tildeapprate * F(lmbda * barapprate)**2
    t1 = -apprate * tildeapprate * lmbda * np.exp(-2 * lmbda * barapprate) / barapprate
    t2 = lmbda * apprate * (1 - apprate) * \
            (-tildeapprate * F(lmbda * barapprate) / barapprate + tildeapprate * np.exp(-lmbda * barapprate) / barapprate) * \
            (a_c * tildeapprate * F(lmbda * barapprate) / barapprate + (1-a_c) * apprate * np.exp(-lmbda * barapprate) / barapprate)
    t3 = lmbda * tildeapprate * (1 - tildeapprate) * \
            ((1-a_c) * apprate * F(lmbda * barapprate) / barapprate + a_c * tildeapprate * np.exp(-lmbda * barapprate) / barapprate) * \
            (-apprate * F(lmbda * barapprate) / barapprate + apprate * np.exp(-lmbda * barapprate) / barapprate)
    return -2 * (t1 + t2 + t3) + cvself_cr

def nvar_cr(conrate, tildeconrate, lmbda, a_c):
    apprate = G(conrate)
    tildeapprate = G(tildeconrate)
    barapprate = (1-a_c) * apprate + a_c * tildeapprate
    return vt_cr(apprate, tildeapprate, barapprate, lmbda, a_c) + \
            vc_cr(apprate, tildeapprate, barapprate, lmbda, a_c) + \
            cvtt_cr(apprate, tildeapprate, barapprate, lmbda, a_c) + \
            cvcc_cr(apprate, tildeapprate, barapprate, lmbda, a_c) + \
            cvtc_cr(apprate, tildeapprate, barapprate, lmbda, a_c)

def finite_var_lr(conrate, tilde_conrate, lmbd, a_l, n):
    return nvar_lr(conrate, tilde_conrate, lmbd, a_l) / n

def finite_var_cr(conrate, tilde_conrate, lmbd, a_c, n):
    return nvar_cr(conrate, tilde_conrate, lmbd, a_c) / n

def next_line(infile):
    while True:
        # Handle comment and empty lines
        _l = next(infile).strip()
        if _l and _l[0] != '#':
            return _l

def get_int(infile):
    return int(next_line(infile))

def get_float(infile):
    return float(next_line(infile))

def get_ks(infile):
    return next_line(infile).split(' ')[1:]

def get_N_ints(n, infile):
    return [int(next_line(infile)) for _ in range(n)]

def get_N_floats(n, infile):
    return [float(next_line(infile)) for _ in range(n)]

os.chdir("./Outputs")
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=12)
plt.rc('axes', labelsize=14)
plt.rc('axes', titlesize=13)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

colorWheel = [
        # '#329932',
        # '#ff6961',
        # '#6a3d9a',
        # '#fb9a99',
        # '#e31a1c',
            '#fdbf6f',
            'b',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928',
            '#67001f',
            '#b2182b',
            '#053061']

def G(x): return 1-np.exp(-x)

def F(x): return G(x) / x

