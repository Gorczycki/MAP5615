#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "Mersenne.h"

using namespace std;

double theta_mersenne(vector<double> vals);
double mean(vector<double> thetas);
double stddev(vector<double> thetas, double mean);

int main(int argc, char * argv[])
{
    int amount = 10000;
    uint32_t seed = stoi(string(argv[1]));
    //cout << seed << endl;
    // uint32_t seed = 1248;
    int runs = 40;
    double exponential_mean;
    double exponential_stddev;

    vector<double> storer; //random values
    double theta; //individual theta estimate
    vector<double> thetas; //estimations
    // decimal precision?
    // double precision (64 bit floating point) is standard
    // double is a 64 bit floating point (8 bytes)
    // adding +1 to seed each iteration, is this an issue?
    // vector<double> thetas working correctly

    //storer = generateMersenne(seed, amount);
    //for(int i = 0; i<storer.size(); i++) //no negatives
    //{
    //    //if(storer[i] <= 0)
    //    //    cout<<"found";
    //    //else
    //    //    cout<<"working";
    //}

    for(int i = 0; i<runs; i++)
    {
        storer = generateMersenne(seed, amount);
        theta = theta_mersenne(storer);
        thetas.push_back(theta);
        seed++;
    }

    exponential_mean = mean(thetas);

    cout<<fixed<<setprecision(10)<<exponential_mean;

    exponential_stddev = stddev(thetas, exponential_mean);

    cout<<endl;
    cout<<exponential_stddev;





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
