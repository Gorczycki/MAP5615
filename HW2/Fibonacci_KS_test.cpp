#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdint.h>

using namespace std;

vector<long long> generator(long long x_n, long long x_n_1);
double KStest(const vector<double>& test);

int main()
{
    //int seed;
    vector<long long> fibonacci;
    long long x_n = 1;
    long long x_n_1 = x_n;
    int amount = 998;
    int x = 0;
    fibonacci.push_back(x_n);
    fibonacci.push_back(x_n_1);
    double KS_stat;
    
    while(x < amount)
    {
        vector<long long> passer = generator(x_n, x_n_1);
        fibonacci.push_back(passer[0]);
        x_n_1 = x_n;
        x_n = passer[0];
        x++;
    }

    vector<double> test; //normalized to (0,1):
    for(long long b : fibonacci)
    {
        double y = static_cast<double>(b) / pow(2,31);
        test.push_back(y);
    }

    sort(test.begin(), test.end());
    
    KS_stat = KStest(test);

    cout<<KS_stat;

    



    return 0;
}

vector<long long> generator(long long x_n, long long x_n_1)
{
    vector<long long> ans = {};
    long long first = x_n + x_n_1;
    long long modulus = 1LL << 31;  // Using bit-shifting for 2^31
    first = first % modulus;
    ans.push_back(first);
    ans.push_back(x_n);
    ans.push_back(x_n_1);



    return ans;
}

double KStest(const vector<double>& test) //pass by const reference
{
    double D_max = 0;
    double D_plus = 0;
    double D_minus = 0;
    double D_N;

    for(int i = 0; i<test.size(); i++)
    {
        double F_i = static_cast<double>(i+1)/test.size();
        D_plus = max(D_plus, abs(F_i - test[i]));
        D_minus = max(D_minus, (abs(test[i] - (static_cast<double>(i))/test.size())));
    }
    D_N = max(D_plus, D_minus);




    return D_N;
}
