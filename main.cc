#include <iostream>   
#include <fstream>
#include <vector>

#include "Marketplace.h"

using namespace std;

int main()
{
    cout.precision(12);

    unsigned long long m = 1'000'000ULL; // 22'674'816ULL;
    unsigned long long n = 1'000'000ULL; // 524'288ULL;
    Marketplace mp(n, m, 0.0001, 0.9999);
    // Marketplace mp(n, m, 0.11133, 0.11258);
    /*
    mp.run_LR();
    cout << mp.get_LR_estimator() << ' ' << mp.get_LR_estimator_nonlinear() << endl;
    mp.run_LR();
    cout << mp.get_LR_estimator() << ' ' << mp.get_LR_estimator_nonlinear() << endl;
    mp.run_LR();
    cout << mp.get_LR_estimator() << ' ' << mp.get_LR_estimator_nonlinear() << endl;
    mp.run_LR();
    cout << mp.get_LR_estimator() << ' ' << mp.get_LR_estimator_nonlinear() << endl;
    mp.run_LR();
    cout << mp.get_LR_estimator() << ' ' << mp.get_LR_estimator_nonlinear() << endl;
    mp.run_LR();
    cout << mp.get_LR_estimator() << ' ' << mp.get_LR_estimator_nonlinear() << endl;
    mp.run_LR();
    cout << mp.get_LR_estimator() << ' ' << mp.get_LR_estimator_nonlinear() << endl;
    mp.run_LR();
    cout << mp.get_LR_estimator() << ' ' << mp.get_LR_estimator_nonlinear() << endl << endl;
    mp.run_CR();
    cout << mp.get_CR_estimator() << ' ' << mp.get_CR_estimator_nonlinear() << endl;
    mp.run_CR();
    cout << mp.get_CR_estimator() << ' ' << mp.get_CR_estimator_nonlinear() << endl;
    mp.run_CR();
    cout << mp.get_CR_estimator() << ' ' << mp.get_CR_estimator_nonlinear() << endl;
    mp.run_CR();
    cout << mp.get_CR_estimator() << ' ' << mp.get_CR_estimator_nonlinear() << endl;
    mp.run_CR();
    cout << mp.get_CR_estimator() << ' ' << mp.get_CR_estimator_nonlinear() << endl;
    mp.run_CR();
    cout << mp.get_CR_estimator() << ' ' << mp.get_CR_estimator_nonlinear() << endl;
    mp.run_CR();
    cout << mp.get_CR_estimator() << ' ' << mp.get_CR_estimator_nonlinear() << endl;
    */
    for (int i = 0; i < 10; i++) {
        mp.run_CR();
        cout << mp.get_CR_estimator() << endl;
        // cout << mp.get_CR_estimator() << ' ' << mp.get_CR_estimator_nonlinear() << endl;
    }
    // mp.run_TSR(0.75, 0.75);
    // cout << mp.get_TSR_estimator() << endl;
    cout << mp.gte << endl;
    // mp.LR_bias_n_times(10);
    // mp.CR_bias_n_times(10);
    // vector<long double> ks = {1.0, 2.0};
    // mp.TSR_bias_n_times(10, 0.75, 0.75, ks, 0.5);
    return 0;
}
