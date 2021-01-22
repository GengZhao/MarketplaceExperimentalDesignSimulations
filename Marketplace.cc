#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "Marketplace.h"

using namespace std;

Marketplace::Marketplace(
        const unsigned long long n,
        const unsigned long long m,
        const vector<long double>& ltype_fractions,
        const vector<long double>& ctype_fractions,
        const vector<vector<long double> >& phi_0,
        const vector<vector<long double> >& phi_1
    ) : rng(rd()), n(n), m(m), phi_0(phi_0), phi_1(phi_1),
    ltype_fractions(ltype_fractions), ctype_fractions(ctype_fractions),
    n_ltypes(ltype_fractions.size()), n_ctypes(ctype_fractions.size()),
    ltype_start_indices(n_ltypes), ctype_start_indices(n_ctypes), 
    ltype_counts(n_ltypes), ctype_counts(n_ctypes),
    edge_prob_0(n_ctypes, vector<long double>(n_ltypes)), edge_prob_1(n_ctypes, vector<long double>(n_ltypes)),
    // empty_consideration_prob_0(pow(1 - edge_prob_0, n)), empty_consideration_prob_1(pow(1 - edge_prob_1, n)),
    gte(0.0)
    // gte(pow(1 - (1 - empty_consideration_prob_0) / n, m) - pow(1 - (1 - empty_consideration_prob_1) / n, m))
{
    long double lfrac = 0.0, cfrac = 0.0;
    unsigned long long ltype_start = 0ULL;
    for (unsigned short ltype = 0; ltype < n_ltypes; ltype++) {
        ltype_start_indices[ltype] = ltype_start;
        lfrac += ltype_fractions[ltype];
        ltype_start = (unsigned long long) round(lfrac * n);
        ltype_counts[ltype] = ltype_start - ltype_start_indices[ltype];
    }
    assert(ltype_start == n);
    unsigned long long ctype_start = 0ULL;
    for (unsigned short ctype = 0; ctype < n_ctypes; ctype++) {
        ctype_start_indices[ctype] = ctype_start;
        cfrac += ctype_fractions[ctype];
        ctype_start = (unsigned long long) round(cfrac * m);
        ctype_counts[ctype] = ctype_start - ctype_start_indices[ctype];
    }
    assert(ctype_start == m);

    for (unsigned short ctype = 0; ctype < n_ctypes; ctype++) {
        transform(phi_0[ctype].begin(), phi_0[ctype].end(), edge_prob_0[ctype].begin(), bind2nd(divides<long double>(), n));
        transform(phi_1[ctype].begin(), phi_1[ctype].end(), edge_prob_1[ctype].begin(), bind2nd(divides<long double>(), n));
    }

    // gte
    vector<vector<long double> > phi_0_star(n_ctypes, vector<long double>(n_ltypes)), phi_1_star(n_ctypes, vector<long double>(n_ltypes));
    for (unsigned short ctype = 0; ctype < n_ctypes; ctype++) {
        long double c_consider_rates_0 = inner_product(ltype_fractions.begin(), ltype_fractions.end(), phi_0[ctype].begin(), (long double) 0.0);
        long double c_consider_rates_1 = inner_product(ltype_fractions.begin(), ltype_fractions.end(), phi_1[ctype].begin(), (long double) 0.0);
        transform(
                phi_0[ctype].begin(),
                phi_0[ctype].end(),
                phi_0_star[ctype].begin(),
                bind1st(multiplies<long double>(),
                    (1 - exp(-c_consider_rates_0)) / c_consider_rates_0 * m / n));
        transform(
                phi_1[ctype].begin(),
                phi_1[ctype].end(),
                phi_1_star[ctype].begin(),
                bind1st(multiplies<long double>(),
                    (1 - exp(-c_consider_rates_1)) / c_consider_rates_1 * m / n));
    }
    vector<long double> l_book_rates_0(n_ltypes), l_book_rates_1(n_ltypes);
    for (unsigned short ltype = 0; ltype < n_ltypes; ltype++) {
        long double l_application_rate_0 = 0.0, l_application_rate_1 = 0.0;
        for (unsigned short ctype = 0; ctype < n_ctypes; ctype++) {
            l_application_rate_0 += phi_0_star[ctype][ltype] * ctype_fractions[ctype];
            l_application_rate_1 += phi_1_star[ctype][ltype] * ctype_fractions[ctype];
        }
        gte += ltype_fractions[ltype] * (exp(-l_application_rate_0) - exp(-l_application_rate_1));
    }
}

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

    map<pair<unsigned short, unsigned long long>, vector<pair<unsigned short, unsigned long long> > > applications;
    vector<unsigned long long> n_1s, n_0s;
    transform(ltype_counts.begin(), ltype_counts.end(), back_inserter(n_1s), bind1st(multiplies<long double>(), a_l));
    transform(ltype_counts.begin(), ltype_counts.end(), n_1s.begin(), back_inserter(n_0s), minus<unsigned long long>());

    for (unsigned short ctype = 0; ctype < n_ctypes; ctype++) {
        unsigned long long ctype_count = ctype_counts[ctype];
        long double m_1 = ctype_count * a_c;
        for (unsigned long long c = 0; c < ctype_count; c++) {
            // TODO
            vector<unsigned long long> n_ltype_treatment_listings_considered(n_ltypes), n_ltype_control_listings_considered(n_ltypes);
            if (c < m_1) { // treatment customer
                for (unsigned short ltype = 0; ltype < n_ltypes; ltype++) {
                    n_ltype_treatment_listings_considered[ltype] = n_1s[ltype] && binomial_distribution<unsigned long long>(n_1s[ltype], edge_prob_1[ctype][ltype])(rng);
                    n_ltype_control_listings_considered[ltype] = n_0s[ltype] && binomial_distribution<unsigned long long>(n_0s[ltype], edge_prob_0[ctype][ltype])(rng);
                    // unsigned long long n_treatment_listing_considered = n_1 && min(poisson_distribution<unsigned long long>(phi_1 * a_l)(rng), n_1);
                    // unsigned long long n_control_listings_considered = n_0 && min(poisson_distribution<unsigned long long>(phi_0 * (1 - a_l))(rng), n_0);
                }
            } else { // control customer
                for (unsigned short ltype = 0; ltype < n_ltypes; ltype++) {
                    n_ltype_treatment_listings_considered[ltype] = n_1s[ltype] && binomial_distribution<unsigned long long>(n_1s[ltype], edge_prob_0[ctype][ltype])(rng);
                    n_ltype_control_listings_considered[ltype] = n_0s[ltype] && binomial_distribution<unsigned long long>(n_0s[ltype], edge_prob_0[ctype][ltype])(rng);
                }
            }
            unsigned long long n_treatment_listings_considered = accumulate(
                    n_ltype_treatment_listings_considered.begin(),
                    n_ltype_treatment_listings_considered.end(),
                    decltype(n_ltype_treatment_listings_considered)::value_type(0.0)
                );
            unsigned long long n_control_listings_considered = accumulate(
                    n_ltype_control_listings_considered.begin(),
                    n_ltype_control_listings_considered.end(),
                    decltype(n_ltype_control_listings_considered)::value_type(0.0)
                );
            unsigned long long total_considered = n_treatment_listings_considered + n_control_listings_considered;
            if (total_considered) {
                unsigned short listing_to_apply_type;
                unsigned long long listing_to_apply_index;
                if (uniform_real_distribution<long double>(0.0, 1.0)(rng) * total_considered < n_treatment_listings_considered) {
                    listing_to_apply_type = discrete_distribution<unsigned short>(
                            n_ltype_treatment_listings_considered.begin(),
                            n_ltype_treatment_listings_considered.end()
                        )(rng);
                    listing_to_apply_index = uniform_int_distribution<unsigned long long>(0, n_1s[listing_to_apply_type] - 1)(rng);
                } else {
                    listing_to_apply_type = discrete_distribution<unsigned short>(
                            n_ltype_control_listings_considered.begin(),
                            n_ltype_control_listings_considered.end()
                        )(rng);
                    listing_to_apply_index = uniform_int_distribution<unsigned long long>(n_1s[listing_to_apply_type], ltype_counts[listing_to_apply_type] - 1)(rng);
                }
                applications[pair<unsigned short, unsigned long long>{listing_to_apply_type, listing_to_apply_index}]
                    .push_back(pair<unsigned short, unsigned long long>{ctype, c});
            }
            // } else { // control customer
                // if (uniform_real_distribution<long double>(0.0, 1.0)(rng) > empty_consideration_prob_0) {
                    // applications[uniform_int_distribution<unsigned long long>(0, n - 1)(rng)].push_back(c);
                // }
            // }
        }
    }

    /*
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
    */

    for (auto const& it : applications) {
        pair<unsigned short, unsigned long long> c_type_index = it.second[uniform_int_distribution<unsigned long long>(0, it.second.size()-1)(rng)];
        if (it.first.second < ltype_counts[it.first.first] * a_l) {
            if (c_type_index.second < ctype_counts[c_type_index.first] * a_c) {
                q_11++;
            } else {
                q_01++;
            }
        } else {
            if (c_type_index.second < ctype_counts[c_type_index.first] * a_c) {
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

