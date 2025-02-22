#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdint.h>

using namespace std;

double A_squared(vector<double> sample, int t);
vector<double> Corput(int amount);
vector<double> V_maxes(vector<double> Corput, int size);    

int main()
{
    double anderson;
    int amount = 4000;
    int window = 4; //t
    vector<double> corput_seq = Corput(amount); //working
    vector<double> maxes = V_maxes(corput_seq, window); //working
    //cout<<maxes.size();
    //after finding V's with size t = 4, perform anderson-darling on each V_1,...,V_1000?
    //double ans = A_squared(maxes, window);
    //cout<<ans;
    double ADS = A_squared(maxes, window);
    cout<<ADS;







    return 0;
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



vector<double> Corput(int amount)
{
    std::vector<double> fractions;

    for (int i = 1; i <= amount; i++)
    {
        double fraction = 0.0;
        double denominator = 2.0;  // Start with 2^-1 (1/2)

        int n = i;
        
        while (n > 0)
        {
            if (n % 2 == 1)
            {
                fraction += 1.0 / denominator;
            }
            n /= 2;
            denominator *= 2;
        }
        fractions.push_back(fraction);
    }

    return fractions;
}

vector<double> V_maxes(vector<double> Corput, int size)
{
    vector<double> ans;
    vector<double> window;

    for(int i = 0; i<Corput.size(); i++)
    {
        window.push_back(Corput[i]);
        if(window.size() == size)
        {
            sort(window.begin(), window.end());
            ans.push_back(window[window.size()-1]);
            window.clear();
        }
    }


    return ans;
}
