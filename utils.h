#ifndef MY_UTILS_H
#define MY_UTILS_H

#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>

std::string getTime()
{
    const int MAXLEN = 80;
    char s[MAXLEN];
    std::time_t t = std::time(0);
    std::strftime(s, MAXLEN, "%Y-%m-%d_%H:%M:%S", std::localtime(&t));
    return std::string(s);
}

template<typename T>
void printVector(const std::vector<T>& vec, std::ostream& os) {
    std::copy(vec.cbegin(), vec.cend(), std::ostream_iterator<T>(os, "\n"));
}

template<typename T>
void printVectorTsv(const std::vector<T>& vec)
{
    for (typename std::vector<T>::const_iterator it = vec.begin(); it != vec.end(); ++it) {
        std::cout << "\t" << *it;
    }
    std::cout << std::endl;
}

template<typename T>
void printVector2DTsv(const std::vector<T>& vec2d)
{
    for (typename std::vector<T>::const_iterator it = vec2d.begin(); it != vec2d.end(); ++it) {
        printVectorTsv(*it);
    }
}

#endif
