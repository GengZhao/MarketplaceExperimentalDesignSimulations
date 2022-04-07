#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from common import *

m = n = 2**24
phi_0 = 0.2525 # 0.2
coeffs = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.001, 1.01, 1.02, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5])

def G(x): return 1-np.exp(-x)

def F(x): return G(x) / x

gtes = G(G(phi_0 * coeffs)) - G(G(phi_0))
LRs = G(phi_0 * coeffs * F(phi_0 * (coeffs * 0.5 + 0.5))) - G(phi_0 * F(phi_0 * (coeffs * 0.5 + 0.5)))
CRs = (G(phi_0 * coeffs) - G(phi_0)) * F(G(phi_0 * coeffs) * 0.5 + G(phi_0) * 0.5)
lr1rb = (G(phi_0 * coeffs * G(phi_0 * (coeffs * 0.1 + 0.9)) / (phi_0 * (coeffs * 0.1 + 0.9))) - G(phi_0 * G(phi_0 * (coeffs * 0.1 + 0.9)) / (phi_0 * (coeffs * 0.1 + 0.9))) - gtes) / gtes
lr9rb = (G(phi_0 * coeffs * G(phi_0 * (coeffs * 0.9 + 0.1)) / (phi_0 * (coeffs * 0.9 + 0.1))) - G(phi_0 * G(phi_0 * (coeffs * 0.9 + 0.1)) / (phi_0 * (coeffs * 0.9 + 0.1))) - gtes) / gtes
cr1rb = ((G(phi_0 * coeffs) - G(phi_0)) * G(G(phi_0 * coeffs) * 0.1 + G(phi_0) * 0.9) / (G(phi_0 * coeffs) * 0.1 + G(phi_0) * 0.9) - gtes) / gtes
cr9rb = ((G(phi_0 * coeffs) - G(phi_0)) * G(G(phi_0 * coeffs) * 0.9 + G(phi_0) * 0.1) / (G(phi_0 * coeffs) * 0.9 + G(phi_0) * 0.1) - gtes) / gtes

plt.plot(gtes, (CRs - gtes) / gtes, color=colorWheel[0])
plt.plot(gtes, (LRs - gtes) / gtes, color=colorWheel[1])
plt.fill_between(gtes, np.minimum(cr1rb, cr9rb), np.maximum(cr1rb, cr9rb), alpha=0.2, color=colorWheel[0])
plt.fill_between(gtes, np.minimum(lr1rb, lr9rb), np.maximum(lr1rb, lr9rb), alpha=0.2, color=colorWheel[1])
plt.ylim(bottom=0)
plt.xlabel(r'GTE')
plt.ylabel('Bias/GTE')
plt.legend(['CR', 'LR', 'CR range', 'LR range'], loc='lower right')
sns.despine()
plt.show()

lr001rb = (G(phi_0 * coeffs * G(phi_0 * (coeffs * 0.001 + 0.999)) / (phi_0 * (coeffs * 0.001 + 0.999))) - G(phi_0 * G(phi_0 * (coeffs * 0.001 + 0.999)) / (phi_0 * (coeffs * 0.001 + 0.999))) - gtes) / gtes
lr999rb = (G(phi_0 * coeffs * G(phi_0 * (coeffs * 0.999 + 0.001)) / (phi_0 * (coeffs * 0.999 + 0.001))) - G(phi_0 * G(phi_0 * (coeffs * 0.999 + 0.001)) / (phi_0 * (coeffs * 0.999 + 0.001))) - gtes) / gtes
cr001rb = ((G(phi_0 * coeffs) - G(phi_0)) * G(G(phi_0 * coeffs) * 0.001 + G(phi_0) * 0.999) / (G(phi_0 * coeffs) * 0.001 + G(phi_0) * 0.999) - gtes) / gtes
cr999rb = ((G(phi_0 * coeffs) - G(phi_0)) * G(G(phi_0 * coeffs) * 0.999 + G(phi_0) * 0.001) / (G(phi_0 * coeffs) * 0.999 + G(phi_0) * 0.001) - gtes) / gtes

crbound = (phi_0 * (coeffs - 1)) ** 2
plt.plot(coeffs, np.abs(cr001rb - cr999rb), color=colorWheel[0])
plt.plot(coeffs, crbound / np.abs(gtes), color=colorWheel[-1], ls=':')
plt.xlabel(r'Treatment parameter $\alpha$')
plt.ylabel('Relative bias differential')

sns.despine()
plt.show()

