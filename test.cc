#include <iostream>
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
    long long i = 0;
    cout << sizeof(unsigned long long) << " " << sizeof(long double) << endl;
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
