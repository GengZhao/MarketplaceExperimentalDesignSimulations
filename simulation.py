#!/usr/bin/env python
'''
Obsolete. Use C++ simulations instead.
'''
import numpy as np
from matplotlib import pyplot as plt

def match_probabilities_for_listings(utility_matrix):
    action_probability = action_probability_matrix(utility_matrix)
    no_applications_probabilities = (np.ones(utility_matrix.shape) - action_probability).prod(axis=1)
    return 1 - no_applications_probabilities

def expected_number_matches(utility_matrix):
    return match_probabilities(utility_matrix).sum()

def action_probability_matrix(utility_matrix):
    consideration_set_sizes = utility_matrix.sum(0)
    return utility_matrix / np.maximum(consideration_set_sizes, np.ones(consideration_set_sizes.shape))

def rand_experiment_market_utility_matrix(n0, n1, m0, m1, p0, p1):
    return np.concatenate([ \
            np.random.binomial(1, p0, [n0, m0+m1]), \
            np.concatenate([ \
                np.random.binomial(1, p0, [n1,m0]), \
                np.random.binomial(1, p1, [n1,m1]) \
            ], axis=1) \
        ])

if __name__ == '__main__':

    lambda0 = 0.5
    lambda1 = 0.6
    gtes = []
    e_gte_hats = []
    var_U1s = []
    var_Uns = []
    cov_U1_U2s = []
    cov_Unminus1_Uns = []
    cov_U1_Uns = []
    var_gte_hats = []
    p0s = [] # These are approximate, shortcut computation
    p1s = []

    # for m, n in zip(np.logspace(6, 20, 15, base=2), np.logspace(4, 18, 15, base=3)):
    for n in np.logspace(4, 20, 17, base=2):
        # for m_over_n in np.logspace(-9, 5, 15, base=2):
        n = int(n)
        # n = 2**16
        n0 = int(n * 7 / 8)
        n1 = int(n / 8)
        m = n * 4
        # m = n * m_over_n
        p = lambda0 / n0
        p_tilde = lambda1 / n1

        p0 = lambda0 * (1 - np.exp(-lambda0 - lambda1)) / (lambda0 + lambda1) / n0 # These are approximate
        q0 = 1 - p0
        p1 = lambda1 * (1 - np.exp(-lambda0 - lambda1)) / (lambda0 + lambda1) / n1
        q1 = 1 - p1
        if n > m:
            e_gte_hat = (1 - np.power(1 - p1, m)) * n / m - (1 - np.power(1 - p0, m)) * n / m
        else:
            e_gte_hat = (1 - np.power(1 - p1, m)) - (1 - np.power(1 - p0, m))
        p0s.append(p0)
        p1s.append(p1)
        e_gte_hats.append(e_gte_hat)

        var_U1 = np.power(q0, m) * (1 - np.power(q0, m))
        var_U1s.append(var_U1)
        var_Un = np.power(q1, m) * (1 - np.power(q1, m))
        var_Uns.append(var_Un)
        cov_U1_U2 = np.power(1 - 2 * p0, m) - np.power(q0, 2 * m)
        cov_U1_U2s.append(cov_U1_U2)
        cov_Unminus1_Un = np.power(1 - 2 * p1, m) - np.power(q1, 2 * m)
        cov_Unminus1_Uns.append(cov_Unminus1_Un)
        cov_U1_Un = np.power(1 - p0 - p1, m) - np.power(q0, m) * np.power(q1, m)
        cov_U1_Uns.append(cov_U1_Un)
        var_gte_hat = var_U1 / n0 + var_Un / n1 + cov_U1_U2 * (n0 - 1) / n0 + cov_Unminus1_Un * (n1 - 1) / n1 - 2 * cov_U1_Un
        var_gte_hats.append(var_gte_hat)

        q_gt = 1 - np.power(1 - (1 - np.power(1 - p_tilde, n)) / n, m)
        q_gc = 1 - np.power(1 - (1 - np.power(1 - p, n)) / n, m)
        gte = (q_gt - q_gc) * max(1, n / float(m))

        gtes.append(gte)
    '''
    res = [[expected_number_matches(np.random.binomial(1, p, [50,50])) for _ in range(500)] for p in np.logspace(-5,0,100,False)]
    res = np.array(res)
    plt.plot(np.logspace(-5,0,100,False), res.mean(axis=1) / 50)
    plt.xscale('log')
    plt.xlabel('Edge probability p')
    plt.ylabel('Fraction of listings (50 in total) with a matched customer')
    plt.show()

    res = [[expected_number_matches(np.random.binomial(1, p, [80,50])) for _ in range(500)] for p in np.logspace(-5,0,100,False)]
    res = np.array(res)
    plt.plot(np.logspace(-5,0,100,False), res.mean(axis=1) / 80)
    plt.xscale('log')
    plt.xlabel('Edge probability p')
    plt.ylabel('Fraction of listings (80 in total) with a matched customer')
    plt.show()

    res = [[expected_number_matches(np.random.binomial(1, p, [50,80])) for _ in range(500)] for p in np.logspace(-5,0,100,False)]
    res = np.array(res)
    plt.plot(np.logspace(-5,0,100,False), res.mean(axis=1) / 50)
    plt.xscale('log')
    plt.xlabel('Edge probability p')
    plt.ylabel('Fraction of listings (50 in total) with a matched customer')
    plt.show()
    '''

 
