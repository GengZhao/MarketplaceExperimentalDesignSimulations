#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <vector>
#include <tuple>
#include <set>
#include <map>
#include <cmath>
#include <random>

using namespace std;

int main()
{
    // long long i = 0;
    cout << fixed << setprecision(20);
    cout << sizeof(double) << " " << sizeof(long double) << endl;
    cout << (double) 0.1 << " " << (long double) 0.1 << endl;
    cout << (double) 1 / 3.0 << " " << (long double) 1 / 3.0 << endl;
    cout << (double) 0.7 << " " << (long double) 0.7 << endl;
    /*
    while (true) {
        random_device rd;
        mt19937 rng(rd());
        long double p = 0.00000000331789255142211909311;
        unsigned long long n = 16777216ULL;
        unsigned long long x = binomial_distribution<unsigned long long>(n, p)(rng);
        if (i % 200 == 199) cout << '*' << flush;
        i++;
    }
    */
    return 0;
}
