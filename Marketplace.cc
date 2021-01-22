#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <tuple>
#include <set>
#include <map>
#include <cmath>
#include <random>

#include "Marketplace.h"

using namespace std;

Marketplace::Marketplace(const unsigned long long n, const unsigned long long m, const long double phi_0, const long double phi_1)
    : rng(rd()), n(n), m(m), phi_0(phi_0), phi_1(phi_1), edge_prob_0(phi_0 / n), edge_prob_1(phi_1 / n),
    empty_consideration_prob_0(pow(1 - edge_prob_0, n)), empty_consideration_prob_1(pow(1 - edge_prob_1, n)),
    gte(pow(1 - (1 - empty_consideration_prob_0) / n, m) - pow(1 - (1 - empty_consideration_prob_1) / n, m)) { }

void Marketplace::reset()
{
    q_00 = 0;
    q_01 = 0;
    q_10 = 0;
    q_11 = 0;
}

void Marketplace::run_LR(const long double a_l)
{ run_TSR(a_l, 1.0); }

void Marketplace::run_CR(const long double a_c)
{ run_TSR(1.0, a_c); }

void Marketplace::run_TSR(const long double a_l, const long double a_c)
{
    reset();
    this->a_l = a_l;
    this->a_c = a_c;

    map<unsigned long long, vector<unsigned long long> > applications;
    unsigned long long n_1 = n * a_l;
    unsigned long long n_0 = n - n_1;

    for (unsigned long long c = 0; c < m; c++) {
        if (c < m * a_c) { // treatment customer
            unsigned long long n_treatment_listing_considered = n_1 && binomial_distribution<unsigned long long>(n_1, edge_prob_1)(rng);
            unsigned long long n_control_listing_considered = n_0 && binomial_distribution<unsigned long long>(n_0, edge_prob_0)(rng);
            // unsigned long long n_treatment_listing_considered = n_1 && min(poisson_distribution<unsigned long long>(phi_1 * a_l)(rng), n_1);
            // unsigned long long n_control_listing_considered = n_0 && min(poisson_distribution<unsigned long long>(phi_0 * (1 - a_l))(rng), n_0);
            unsigned long long total_considered = n_treatment_listing_considered + n_control_listing_considered;
            if (total_considered) {
                unsigned long long listing_to_apply;
                if (uniform_real_distribution<long double>(0.0, 1.0)(rng) * total_considered < n_treatment_listing_considered) {
                    listing_to_apply = uniform_int_distribution<unsigned long long>(0, n_1 - 1)(rng);
                } else {
                    listing_to_apply = uniform_int_distribution<unsigned long long>(n_1, n - 1)(rng);
                }
                applications[listing_to_apply].push_back(c);
            }
        } else { // control customer
            if (uniform_real_distribution<long double>(0.0, 1.0)(rng) > empty_consideration_prob_0) {
                applications[uniform_int_distribution<unsigned long long>(0, n - 1)(rng)].push_back(c);
            }
        }
    }
    for (auto const& it : applications) {
        unsigned long long c = it.second[uniform_int_distribution<unsigned long long>(0, it.second.size()-1)(rng)];
        if (it.first < n * a_l) {
            if (c < m * a_c) {
                q_11++;
            } else {
                q_01++;
            }
        } else {
            if (c < m * a_c) {
                q_10++;
            } else {
                q_00++;
            }
        }
    }
}

const long double Marketplace::get_LR_estimator()
{
    return q_11 / a_c / a_l / n - q_10 / a_c / (1 - a_l) / n;
}

const long double Marketplace::get_LR_estimator_nonlinear()
{
    long double q_10_normal_complement = 1 - q_10 / a_c / (1 - a_l) / (n + 1); // to prevent zero
    long double q_11_normal_complement = 1 - q_11 / a_c / a_l / (n + 1); // to prevent zero
    long double rho = (long double) m / n;
    long double wgt_log_q_complement = (1 - a_l) * log(q_10_normal_complement) + a_l * log(q_11_normal_complement);
    return exp(-rho * (1 - pow(1 + wgt_log_q_complement / rho, log(q_10_normal_complement) / wgt_log_q_complement))) -
        exp(-rho * (1 - pow(1 + wgt_log_q_complement / rho, log(q_11_normal_complement) / wgt_log_q_complement)));
}

const long double Marketplace::get_CR_estimator()
{
    return q_11 / a_c / a_l / n - q_01 / (1 - a_c) / a_l / n;
}

const long double Marketplace::get_CR_estimator_nonlinear()
{
    return pow(1 - (long double) q_11 / n - (long double) q_01 / n, q_01 / (1 - a_c) / a_l / (q_11 + q_01)) -
        pow(1 - (long double) q_11 / n - (long double) q_01 / n, q_11 / (1 - a_c) / a_l / (q_11 + q_01));
}

const long double Marketplace::get_TSR_estimator()
{
    return q_11 / a_c / a_l / n - (q_10 + q_01 + q_00) / (1 - a_c * a_l) / n;
}

const long double Marketplace::get_TSR_estimator_improved(const long double k, const long double beta)
{
    return q_11 / a_c / a_l / n - 
        beta * (1 - k * (1 - beta)) * q_01 / (1 - a_c) / a_l / n -
        (1 - beta) * (1 - k * beta) * q_10 / a_c / (1 - a_l) / n -
        2 * k * beta * (1 - beta) * q_00 / (1 - a_c) / (1 - a_l) / n;
}

tuple<vector<long double>, vector<long double> > Marketplace::LR_bias_n_times(const int n_it, const long double a_l)
{
    vector<long double> res, res_nl;
    for (int i = 0; i < n_it; i++) {
        run_LR(a_l);
        res.push_back(get_LR_estimator() - gte);
        res_nl.push_back(get_LR_estimator_nonlinear() - gte);
    }
    return tuple<vector<long double>, vector<long double> >{res, res_nl};
}

tuple<vector<long double>, vector<long double> > Marketplace::CR_bias_n_times(const int n_it, const long double a_c)
{
    vector<long double> res, res_nl;
    for (int i = 0; i < n_it; i++) {
        run_CR(a_c);
        res.push_back(get_CR_estimator() - gte);
        res_nl.push_back(get_CR_estimator_nonlinear() - gte);
    }
    return tuple<vector<long double>, vector<long double> >{res, res_nl};
}

vector<vector<long double> > Marketplace::TSR_bias_n_times(const int n_it, const long double a_l, const long double a_c, const vector<long double> ks, const long double beta)
{
    int n_ks = ks.size();
    vector<vector<long double> > res(1 + n_ks);
    for (int i = 0; i < n_it; i++) {
        run_TSR(a_l, a_c);
        res[0].push_back(get_TSR_estimator() - gte);
        for (int ki = 0; ki < n_ks; ki++) {
            res[ki + 1].push_back(get_TSR_estimator_improved(ks[ki], beta) - gte);
        }
    }
    return res;
}

