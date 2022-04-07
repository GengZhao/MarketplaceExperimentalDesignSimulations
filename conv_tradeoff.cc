/* Scale up market size and observe B-V tradeoff in LR/CR
 * Essentially seeing bias stablizing and variance going to zero
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

static void printKs(const vector<long double>& ks, ostream& os) {
    os << ks.size();
    for (auto& k : ks) os << ' ' << k;
    os << '\n';
}

int main() {
    // TODO: Edit these settings before execution
    int nIters = 500;
    int nThrds = 49;
    int nSteps = 10;

	 /* phi     | booking rate = 1 - exp(-(1 - exp(-phi))) in balanced market
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
    long double phi_0 = 0.0419; // 0.28 under 16:1
    long double phi_1 = 0.0456; // 0.3
    unsigned long long n = 2'097'152ULL;
    unsigned long long m = 16'777'216ULL;
    long double base_a_l = 0.5;
    long double base_a_c = 0.5;
    vector<long double> ks = {};

    ofstream outFile;
    string timeStr = getTime();
    outFile.open("bv_tradeoff-" + timeStr + ".txt");

    // Print nSteps
    outFile << nSteps << '\n';

    for (int s = 0; s < nSteps; s++) {
        long double rho = (long double) m / n;
        // Print n, m, phi_0, phi_1, rho
        outFile << n << ' ' << m << ' ' << phi_0 << ' ' << phi_1 << ' ' << rho << '\n';
        // Print number of ks in TSRI and **ks
        printKs(ks, outFile);
        // Print nIters
        outFile << nIters << '\n';

        vector<long double> lr, lr_nl, lr_q10, lr_q11;
        vector<long double> cr, cr_nl, cr_q01, cr_q11;
        // vector<long double> tsr, tsr_q00, tsr_q01, tsr_q10, tsr_q11;
        // vector<vector<long double> > tsris(ks.size());
        lr.reserve(nIters);
        lr_nl.reserve(nIters);
        lr_q10.reserve(nIters);
        lr_q11.reserve(nIters);
        cr.reserve(nIters);
        cr_nl.reserve(nIters);
        cr_q01.reserve(nIters);
        cr_q11.reserve(nIters);
        /*
        tsr.reserve(nIters);
        tsr_q00.reserve(nIters);
        tsr_q01.reserve(nIters);
        tsr_q10.reserve(nIters);
        */
        // tsr_q11.reserve(nIters);
        // for (auto& v : tsris) v.reserve(nIters);

        mutex lockLR, lockCR, lockTSR, lockCounter;
        vector<thread> thrds;
        int counter = 0;

        for (int thrd = 0; thrd < nThrds; thrd++) {
            thrds.push_back(thread([&](int ti) {
                // Initialize instance of Marketplace
                /*
                Marketplace mp(
                        n, m,
                        vector<long double>{0.4, 0.6},
                        vector<long double>{0.3, 0.7},
                        vector<vector<long double> >{vector<long double>{phi_0, phi_0 * 2.5}, vector<long double>{phi_0 * 0.1, phi_0 * 1.6}},
                        vector<vector<long double> >{vector<long double>{phi_1, phi_1 * 2.5}, vector<long double>{phi_1 * 0.1, phi_1 * 1.6}}
                    );
                    */
                Marketplace mp(
                        n, m,
                        vector<long double>{1.0},
                        vector<long double>{0.4, 0.6},
                        vector<vector<long double> >{vector<long double>{phi_0 * 1.5}, vector<long double>{phi_0 * 0.4}},
                        vector<vector<long double> >{vector<long double>{phi_1 * 1.5}, vector<long double>{phi_1 * 0.4}}
                    );
                if (!ti) {
                    outFile << mp.gte << '\n';
                }
                while (true) {
                    lockCounter.lock();
                    if (counter == nIters) {
                        lockCounter.unlock();
                        break;
                    }
                    counter++;
                    if (counter % 20 == 0) cout << '*' << flush;
                    lockCounter.unlock();

                    mp.run_LR(base_a_l);
                    lockLR.lock();
                    lr.push_back(mp.get_LR_estimator());
                    lr_nl.push_back(mp.get_LR_estimator_nonlinear());
                    lr_q10.push_back(mp.get_q10());
                    lr_q11.push_back(mp.get_q11());
                    lockLR.unlock();

                    mp.run_CR(base_a_c);
                    lockCR.lock();
                    cr.push_back(mp.get_CR_estimator());
                    cr_nl.push_back(mp.get_CR_estimator_nonlinear());
                    cr_q01.push_back(mp.get_q01());
                    cr_q11.push_back(mp.get_q11());
                    lockCR.unlock();

                    /*
                    long double beta = exp(-rho);
                    long double a_l = base_a_l * (1 - beta) + beta;
                    long double a_c = 1 - beta + base_a_c * beta;
                    mp.run_TSR(a_l, a_c);
                    lockTSR.lock();
                    tsr.push_back(mp.get_TSR_estimator());
                    tsr_q00.push_back(mp.get_q00());
                    tsr_q01.push_back(mp.get_q01());
                    tsr_q10.push_back(mp.get_q10());
                    tsr_q11.push_back(mp.get_q11());
                    for (int ki = 0; ki < ks.size(); ki++) tsris[ki].push_back(mp.get_TSR_estimator_improved(ks[ki], beta));
                    lockTSR.unlock();
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
        printVector(cr, outFile);
        printVector(cr_nl, outFile);
        printVector(cr_q01, outFile);
        printVector(cr_q11, outFile);
        // printVector(tsr, outFile);
        // for (auto& v : tsris) printVector(v, outFile);
        // printVector(tsr_q00, outFile);
        // printVector(tsr_q01, outFile);
        // printVector(tsr_q10, outFile);
        // printVector(tsr_q11, outFile);

        cout << "=> n & m: " << n << ' ' << m << endl;
        // cout << "=> coeff: " << coeffs[s] << endl;
        n /= 2;
        m /= 2;
    }
    outFile.close();
    return 0;
}
