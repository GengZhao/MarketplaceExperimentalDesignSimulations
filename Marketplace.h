#ifndef MARKETPLACE_H
#define MARKETPLACE_H

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

class Marketplace
{
    private:
        std::random_device rd;
        std::mt19937 rng;

        const unsigned long long n;
        const unsigned long long m;
        std::vector<std::vector<long double> > phi_0;
        std::vector<std::vector<long double> > phi_1;
        std::vector<long double> ltype_fractions;
        std::vector<long double> ctype_fractions;
        int n_ltypes, n_ctypes;
        std::vector<unsigned long long> ltype_start_indices;
        std::vector<unsigned long long> ctype_start_indices;
        std::vector<unsigned long long> ltype_counts;
        std::vector<unsigned long long> ctype_counts;

        long double a_c;
        long double a_l;
        std::vector<std::vector<long double> > edge_prob_0;
        std::vector<std::vector<long double> > edge_prob_1;
        // std::vector<std::vector<long double> > empty_consideration_prob_0;
        // std::vector<std::vector<long double> > empty_consideration_prob_1;

        unsigned long long q_00;
        unsigned long long q_01;
        unsigned long long q_10;
        unsigned long long q_11;

        void reset();

    public:
        long double gte;

        Marketplace(
                const unsigned long long n,
                const unsigned long long m,
                const std::vector<long double>& ltype_fractions,
                const std::vector<long double>& ctype_fractions,
                const std::vector<std::vector<long double> >& phi_0,
                const std::vector<std::vector<long double> >& phi_1
            );

        void run_LR(const long double a_l=0.5);
        void run_CR(const long double a_c=0.5);
        void run_TSR(const long double a_l=0.5, const long double a_c=0.5);

        unsigned long long get_q00() { return q_00; }
        unsigned long long get_q01() { return q_01; }
        unsigned long long get_q10() { return q_10; }
        unsigned long long get_q11() { return q_11; }

        const long double get_LR_estimator();
        const long double get_LR_estimator_nonlinear();
        const long double get_CR_estimator();
        const long double get_CR_estimator_nonlinear();
        const long double get_TSR_estimator();
        const long double get_TSR_estimator_improved(const long double k=1, const long double beta=0.5);

        std::tuple<std::vector<long double>, std::vector<long double> > LR_bias_n_times(const int n_it, const long double a_l=0.5);
        std::tuple<std::vector<long double>, std::vector<long double> > CR_bias_n_times(const int n_it, const long double a_c=0.5);
        std::vector<std::vector<long double> > TSR_bias_n_times(const int n_it, const long double a_l=0.5, const long double a_c=0.5, const std::vector<long double> ks=std::vector<long double>(), const long double beta=0.5);
};

#endif

