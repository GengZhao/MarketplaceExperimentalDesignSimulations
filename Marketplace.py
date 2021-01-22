import numpy as np

class Marketplace(object):

    def __init__(self, n, m, phi_0, phi_1):
        self.n = n
        self.m = m
        self.phi_0 = float(phi_0)
        self.phi_1 = float(phi_1)
        self.a_l = None
        self.a_c = None
        self.edge_prob_0 = self.phi_0 / n
        self.edge_prob_1 = self.phi_1 / n
        self.empty_consideration_prob_0 = (1 - self.edge_prob_0) ** self.n
        self.empty_consideration_prob_1 = (1 - self.edge_prob_1) ** self.n
        self.GTE = (1 - (1 - self.empty_consideration_prob_0) / self.n) ** self.m - \
                (1 - (1 - self.empty_consideration_prob_1) / self.n) ** self.m
        # TODO

    def reset(self):
        self.a_l = None
        self.a_c = None
        self.q_00 = 0
        self.q_01 = 0
        self.q_10 = 0
        self.q_11 = 0

    def run_LR(self, a_l = 0.5):
        self.run_TSR(a_l = a_l, a_c = 1.0)

    def run_CR(self, a_c = 0.5):
        self.run_TSR(a_l = 1.0, a_c = a_c)

    def run_TSR(self, a_l = 0.5, a_c = 0.5):
        self.reset()
        a_l = float(a_l)
        a_c = float(a_c)
        self.a_l = a_l
        self.a_c = a_c

        applications = {}
        n_1 = int(self.n * a_l)
        n_0 = self.n - n_1
        for c in range(int(self.m)):
            if c < self.m * a_c: # treatment customer
                n_treatment_listing_considered = np.random.binomial(n_1, self.edge_prob_1)
                n_control_listing_considered = np.random.binomial(n_0, self.edge_prob_0)
                total_considered = n_treatment_listing_considered + n_control_listing_considered
                if total_considered:
                    if np.random.random() * total_considered < n_treatment_listing_considered:
                        listing_to_apply = int(np.random.random() * n_1)
                    else:
                        listing_to_apply = int(np.random.random() * n_0) + n_1
                    if listing_to_apply not in applications:
                        applications[listing_to_apply] = [c]
                    else:
                        applications[listing_to_apply].append(c)
            else: # control customer
                if np.random.random() > self.empty_consideration_prob_0:
                    listing_to_apply = int(np.random.random() * self.n)
                    if listing_to_apply not in applications:
                        applications[listing_to_apply] = [c]
                    else:
                        applications[listing_to_apply].append(c)
        for l, applicants in applications.items():
            c = np.random.choice(applicants)
            if l < self.n * a_l:
                if c < self.m * a_c:
                    self.q_11 += 1
                else:
                    self.q_01 += 1
            else:
                if c < self.m * a_c:
                    self.q_10 += 1
                else:
                    self.q_00 += 1

    def get_LR_estimator(self):
        return self.q_11 / self.a_c / self.a_l / self.n - self.q_10 / self.a_c / (1 - self.a_l) / self.n

    def get_LR_estimator_nonlinear(self):
        ''' WARNING: not TSR proof'''
        q_10_normal_complement = 1 - self.q_10 / self.a_c / (1 - self.a_l) / (self.n + 1) # to prevent zero
        q_11_normal_complement = 1 - self.q_11 / self.a_c / self.a_l / (self.n + 1) # to prevent zero
        rho = float(self.m) / self.n
        wgt_log_q_complement = (1 - self.a_l) * np.log(q_10_normal_complement) + self.a_l * np.log(q_11_normal_complement)
        return np.exp(-rho * (1 - (1 + wgt_log_q_complement / rho) ** (np.log(q_10_normal_complement) / wgt_log_q_complement))) - \
                np.exp(-rho * (1 - (1 + wgt_log_q_complement / rho) ** (np.log(q_11_normal_complement) / wgt_log_q_complement)))

    def get_CR_estimator(self):
        return self.q_11 / self.a_c / self.a_l / self.n - self.q_01 / (1 - self.a_c) / self.a_l / self.n

    def get_CR_estimator_nonlinear(self):
        ''' WARNING: not TSR proof'''
        return (1 - self.q_11 / float(self.n) - self.q_01 / float(self.n)) ** (self.q_01 / (1 - self.a_c) / self.a_l / (self.q_11 + self.q_01)) - \
                (1 - self.q_11 / float(self.n) - self.q_01 / float(self.n)) ** (self.q_11 / (1 - self.a_c) / self.a_l / (self.q_11 + self.q_01))

    def get_TSR_estimator(self):
        return self.q_11 / self.a_c / self.a_l / self.n - (self.q_01 + self.q_10 + self.q_00) / (1 - self.a_c * self.a_l) / self.n

    def get_TSR_estimator_improved(self, k = 1, beta = 0.5):
        return self.q_11 / self.a_c / self.a_l / self.n - \
                beta * (1 - k * (1 - beta)) * self.q_01 / (1 - self.a_c) / self.a_l / self.n - \
                (1 - beta) * (1 - k * beta) * self.q_10 / self.a_c / (1 - self.a_l) / self.n - \
                2 * k * beta * (1 - beta) * self.q_00 / (1 - self.a_c) / (1 - self.a_l) / self.n

    def LR_bias_n_times(self, n, a_l = 0.5):
        res = []
        res_nl = []
        for i in range(n):
            self.run_LR(a_l = a_l)
            res.append(self.get_LR_estimator() - self.GTE)
            res_nl.append(self.get_LR_estimator_nonlinear() - self.GTE)
            if i % 100 == 0: print('LR time:', i)
        return res, res_nl

    def CR_bias_n_times(self, n, a_c = 0.5):
        res = []
        res_nl = []
        for i in range(n):
            self.run_CR(a_c = a_c)
            res.append(self.get_CR_estimator() - self.GTE)
            res_nl.append(self.get_CR_estimator_nonlinear() - self.GTE)
            if i % 100 == 0: print('CR time:', i)
        return res, res_nl

    def TSR_bias_n_times(self, n, a_l = 0.5, a_c = 0.5, beta = 0.5, ks = []):
        res = []
        res_imp = {k : [] for k in ks}
        for i in range(n):
            self.run_TSR(a_l = a_l, a_c = a_c)
            res.append(self.get_TSR_estimator() - self.GTE)
            for k in ks:
                res_imp[k].append(self.get_TSR_estimator_improved(k = k, beta = beta) - self.GTE)
            if i % 100 == 0: print('TSR time:', i)
        return [res] + list(res_imp.values())

