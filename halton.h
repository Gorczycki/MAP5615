#ifndef HALTON_H
#define HALTON_H
#include <vector>

std::vector<double> generate_halton(int start, int window, int seed);
//start must start with 1 not 0
//seed increased by 1 each time

#endif 
