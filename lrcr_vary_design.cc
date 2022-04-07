/* Vary allocation ratio in LR/CR experiments.
 * Log market characteristics and
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cmath>
#include <thread>
#include <mutex>
#include <string>

#include "Marketplace.h"
#include "utils.h"

using namespace std;

int main() {
    // TODO: Edit these settings before execution
    int nIters = 10000;
    int nThrds = 48;

    /* phi     | booking rate = 1 - exp(-(1 - exp(-phi)))
     * --------+--------
     * 0.11133 | 0.1
     * 0.11258 | 0.101
     * 0.1239  | 0.11
     * 0.17736 | 0.15
     * 0.2525  | 0.2
     * 0.28563 | 0.22
     * 0.33924 | 0.25
     * 0.44111 | 0.3
     */
    // TODO: Edit these settings before execution
    long double phi_0 = 0.2525;
    long double phi_1 = 0.33924;
    unsigned long long n = 1'000'000ULL;
    unsigned long long m = 1'000'000ULL;
    long double rho = (long double) m / n;

    ofstream outFile;
    string timeStr = getTime();
    outFile.open("lrcr_vary_design-" + timeStr + ".txt"); // Output file name

    long double stepSize = 0.4; // TODO: modify step size for a_l and a_c
    long double a_l = 1.0;
    long double a_c = 0.1;

    // First run CR, scanning through values of a_c
    while (a_c <= 0.99) {
        // Print market characteristics n, m, phi_0, phi_1, rho, a_l, a_c
        outFile << n << ' ' << m << ' ' << phi_0 << ' ' << phi_1 << ' ' << rho << ' ' << a_l << ' ' << a_c << '\n';
        // Print nIters
        outFile << nIters << '\n';

        // vector<long double> lr, lr_nl, lr_q10, lr_q11;
        vector<long double> cr, cr_nl, cr_q01, cr_q11;
        //
        // lr.reserve(nIters);
        // lr_nl.reserve(nIters);
        // lr_q10.reserve(nIters);
        // lr_q11.reserve(nIters);
        
        cr.reserve(nIters);
        cr_nl.reserve(nIters);
        cr_q01.reserve(nIters);
        cr_q11.reserve(nIters);

        mutex lockLR, lockCR, lockTSR, lockCounter;
        vector<thread> thrds;
        int counter = 0;

        for (int thrd = 0; thrd < nThrds; thrd++) {
            thrds.push_back(thread([&](int ti) {
                // Initialize instance of Marketplace
                Marketplace mp(
                        n, m,
                        vector<long double>{1.0},
                        vector<long double>{1.0},
                        vector<vector<long double> >{vector<long double>{phi_0}},
                        vector<vector<long double> >{vector<long double>{phi_1}}
                    );
                while (true) {
                    lockCounter.lock();
                    if (counter == nIters) {
                        lockCounter.unlock();
                        break;
                    }
                    counter++;
                    if (counter % 100 == 0) cout << '*' << flush;
                    lockCounter.unlock();

                    // mp.run_LR(base_a_l);
                    // lockLR.lock();
                    // lr.push_back(mp.get_LR_estimator());
                    // lr_nl.push_back(mp.get_LR_estimator_nonlinear());
                    // lr_q10.push_back(mp.get_q10());
                    // lr_q11.push_back(mp.get_q11());
                    // lockLR.unlock();

                    mp.run_CR(a_c);
                    lockCR.lock();
                    cr.push_back(mp.get_CR_estimator());
                    cr_nl.push_back(mp.get_CR_estimator_nonlinear());
                    cr_q01.push_back(mp.get_q01());
                    cr_q11.push_back(mp.get_q11());
                    lockCR.unlock();
                }
            }, thrd));
        }
        for (thread& thrd : thrds) thrd.join();
        cout << endl;

        // printVector(lr, outFile);
        // printVector(lr_nl, outFile);
        // printVector(lr_q10, outFile);
        // printVector(lr_q11, outFile);

        printVector(cr, outFile);
        printVector(cr_nl, outFile);
        printVector(cr_q01, outFile);
        printVector(cr_q11, outFile);

        cout << "=> a_l & a_c: " << a_l << ' ' << a_c << endl;
        a_c += stepSize;
    }

    a_l = 0.1;
    a_c = 1.0;
    // Next run LR, scanning through values of a_l
    while (a_l <= 0.99) {
        // Print market characteristics n, m, phi_0, phi_1, rho, a_l, a_c
        outFile << n << ' ' << m << ' ' << phi_0 << ' ' << phi_1 << ' ' << rho << ' ' << a_l << ' ' << a_c << '\n';
        // Print nIters
        outFile << nIters << '\n';

        vector<long double> lr, lr_nl, lr_q10, lr_q11;
        // vector<long double> cr, cr_nl, cr_q01, cr_q11;
        lr.reserve(nIters);
        lr_nl.reserve(nIters);
        lr_q10.reserve(nIters);
        lr_q11.reserve(nIters);
        /*
        cr.reserve(nIters);
        cr_nl.reserve(nIters);
        cr_q01.reserve(nIters);
        cr_q11.reserve(nIters);
        */

        mutex lockLR, lockCR, lockTSR, lockCounter;
        vector<thread> thrds;
        int counter = 0;

        for (int thrd = 0; thrd < nThrds; thrd++) {
            thrds.push_back(thread([&](int ti) {
                // Initialize instance of Marketplace
                Marketplace mp(
                        n, m,
                        vector<long double>{1.0},
                        vector<long double>{1.0},
                        vector<vector<long double> >{vector<long double>{phi_0}},
                        vector<vector<long double> >{vector<long double>{phi_1}}
                    );
                while (true) {
                    lockCounter.lock();
                    if (counter == nIters) {
                        lockCounter.unlock();
                        break;
                    }
                    counter++;
                    if (counter % 100 == 0) cout << '*' << flush;
                    lockCounter.unlock();

                    mp.run_LR(a_l);
                    lockLR.lock();
                    lr.push_back(mp.get_LR_estimator());
                    lr_nl.push_back(mp.get_LR_estimator_nonlinear());
                    lr_q10.push_back(mp.get_q10());
                    lr_q11.push_back(mp.get_q11());
                    lockLR.unlock();

                    /*
                    mp.run_CR(base_a_c);
                    lockCR.lock();
                    cr.push_back(mp.get_CR_estimator());
                    cr_nl.push_back(mp.get_CR_estimator_nonlinear());
                    cr_q01.push_back(mp.get_q01());
                    cr_q11.push_back(mp.get_q11());
                    lockCR.unlock();
                    */
                }
            }, thrd));
        }
        for (thread& thrd : thrds) thrd.join();
        cout << endl;

        printVector(lr, outFile);
        printVector(lr_nl, outFile);
        printVector(lr_q10, outFile);
        printVector(lr_q11, outFile);
        /*
        printVector(cr, outFile);
        printVector(cr_nl, outFile);
        printVector(cr_q01, outFile);
        printVector(cr_q11, outFile);
        */

        cout << "=> a_l & a_c: " << a_l << ' ' << a_c << endl;

        a_l += stepSize;
    }

    outFile.close();
    return 0;
}
