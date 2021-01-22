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
    int nIters = 20000;
    int nThrds = 48;
    int nSteps = 14;

    long double phi_0 = 0.11133; // 0.11133;
    long double phi_1 = 0.14997; // 0.11258;
    unsigned long long n = 16'777'216ULL; // 536'870'912ULL; // 2'097'152ULL; // 1'048'576ULL; // 2^20
    unsigned long long m = 16'777'216ULL; // 67'108'864ULL; // 16'777'216ULL; // 90'699'264ULL; // 3^11 * 2^9
    long double base_a_l = 0.5;
    long double base_a_c = 0.5;
    vector<long double> ks = {1.0, 2.0};

    ofstream outFile;
    string timeStr = getTime();
    outFile.open("eval_estimators-" + timeStr + ".txt");

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
        vector<long double> tsr, tsr_q00, tsr_q01, tsr_q10, tsr_q11;
        vector<vector<long double> > tsris(ks.size());
        lr.reserve(nIters);
        lr_nl.reserve(nIters);
        lr_q10.reserve(nIters);
        lr_q11.reserve(nIters);
        cr.reserve(nIters);
        cr_nl.reserve(nIters);
        cr_q01.reserve(nIters);
        cr_q11.reserve(nIters);
        tsr.reserve(nIters);
        tsr_q00.reserve(nIters);
        tsr_q01.reserve(nIters);
        tsr_q10.reserve(nIters);
        tsr_q11.reserve(nIters);
        for (auto& v : tsris) v.reserve(nIters);

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
                    if (counter % 200 == 0) cout << '*' << flush;
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
        printVector(tsr, outFile);
        for (auto& v : tsris) printVector(v, outFile);
        printVector(tsr_q00, outFile);
        printVector(tsr_q01, outFile);
        printVector(tsr_q10, outFile);
        printVector(tsr_q11, outFile);

        cout << "=> n & m: " << n << ' ' << m << endl;
        n /= 2;
        m /= 2;
    }
    outFile.close();
    return 0;
}
