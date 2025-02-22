#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdint.h>


using namespace std;

vector<double> haltonandmax(int amount, vector<int> numbases);
double A_squared(vector<double> sample, int t);

int main()
{
    //primes bases: 2,3,5,7
    //i-th halton vector contains a num from each bases [n_b_1, n_b_2, n_b_3, n_b_4]
    vector<int> numbases = {2,3,5,7};
    int amount = 4000;
    vector<double> answer = haltonandmax(amount, numbases);
    int t = 4;
    double ans = A_squared(answer, t);

    cout<<ans;

    
    




    return 0;
}

vector<double> haltonandmax(int amount, vector<int> numbases) {
    vector<double> max_values;

    for (int i = 1; i <= amount; i += 4) { // Only store every 4th value
        vector<double> point;
        
        for (int base : numbases) {
            double fraction = 0.0;
            double denominator = 1.0;
            int n = i; 

            while (n > 0) {
                denominator *= base;
                fraction += (n % base) / denominator;
                n /= base;
            }
            
            point.push_back(fraction);
        }

        double max_value = *max_element(point.begin(), point.end());
        max_values.push_back(max_value);
    }

    return max_values;
}


double A_squared(vector<double> sample, int t)
{
    sort(sample.begin(), sample.end());
    double A;
    double sum = 0;
    int N = sample.size();
    for(int i = 0; i<sample.size(); i++)
    {
        double F_x_i = pow(sample[i],t);
        double F_x_n_i = pow(sample[sample.size()-i-1],t);
        sum += (2 * i + 1) * log(F_x_i) + (N*2 + 1 - 2*i)*(log(1 - F_x_n_i));
    }

    sum = (-1.0*sum) / (sample.size());
    sum -= sample.size();
    A = sum;


    
    return A;
}
