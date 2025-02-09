#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "halton.h"

using namespace std;

double theta_mersenne(vector<double> vals);
double mean(vector<double> thetas);
double stddev(vector<double> thetas, double mean);

int main()
{
    int start = 1;
    int window = 10000;
    int seed = 16344;
    int runs = 40;
    double exponential_mean;
    double exponential_stddev;

    //outcome dependent on window

    //and then shuffle the order in which each corput sequence comes in:
    vector<double> storer; //random values
    double theta; //individual theta estimate
    vector<double> thetas; //estimations

    for(int i = 0; i<runs; i++)
    {
        storer = generate_halton(start, window, seed);
        theta = theta_mersenne(storer);
        thetas.push_back(theta);
        start += 10000;
        window += 10000;
        seed++;
    }

    exponential_mean = mean(thetas);

    cout<<fixed<<setprecision(10)<<exponential_mean;

    exponential_stddev = stddev(thetas, exponential_mean);

    cout<<endl;
    cout<<fixed<<setprecision(10)<<exponential_stddev;




    return 0;
}


double theta_mersenne(vector<double> vals)
{
    double ans = 0;
    for(int i = 0; i<vals.size(); i++)
        ans = ans + exp(vals[i]);

    ans = ans / vals.size();

    return ans;
}

double mean(vector<double> thetas)
{
    double sum = 0;
    double mean = 0;
    for(int i = 0; i<thetas.size(); i++)
    {
        sum += thetas[i];
    }
    mean = sum / thetas.size();


    return mean;
}

double stddev(vector<double> thetas, double mean)
{
    double ans;

    // E[x]^2:
    double sum1 = 0;
    double mean1 = 0;
    for(int i = 0; i<thetas.size(); i++)
    {
        thetas[i] = pow(thetas[i], 2);
        sum1 += thetas[i];
    }
    mean1 = sum1 / thetas.size();

    //(E[x])^2:
    double mean2 = pow(mean,2);

    //ans:
    double var = abs(mean1 - mean2);
    double sqrt_var = sqrt(var);
    ans = sqrt_var;

    return ans;
}   
