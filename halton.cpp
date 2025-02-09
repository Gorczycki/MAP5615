#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <stdint.h>
#include <math.h>
#include "halton.h"

std::vector<double> generate_halton(int start, int window, int seed)
{
    //permuted van der corput, not halton!

    //base 2
    std::vector<std::vector<int> > binaries;
    std::vector<int> submitter;
    std::vector<double> fractions;
    double fraction = 0;

    //first (window) numbers:

    for(int i = start; i<=window; i++)
    {
        int n = i;
        while(n > 0)
        {
            submitter.push_back(n % 2);
            n /= 2;
        }
        binaries.push_back(submitter);
        submitter.clear();
    }

    //make fractions:

    for(int i = 0; i<binaries.size(); i++)
    {
        for(int j = 0; j<binaries[i].size(); j++)
        {
            if(binaries[i][j] == 1)
            {
                fraction += 1/pow(2,j+1);
            }
        }
        fractions.push_back(fraction);
        fraction = 0;
    }

    //now permute or randomize:
    //will use simple c++ shuffle()
    std::default_random_engine rng(seed);
    std::shuffle(fractions.begin(), fractions.end(), rng);

    binaries.clear();
    submitter.clear();




    return fractions;
}
