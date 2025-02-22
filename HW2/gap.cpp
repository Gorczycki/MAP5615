#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdint.h>
#include <iomanip>
#include <map>

std::vector<double> generator(int amount, int k_1, int k_2);
std::map<int, int> gaptest(std::vector<double> ans, int val, double j);

int main()
{
    //x_n \equiv x_{n - k_1} + x_{n-k_2} (mod 2^35)
    //x_n \equiv x_{n - 10} + x_{n - 16} (mod 2^35)
    //x_16 \equiv x_{16 - 10} + x_{16-16} = x_6 + x_0
    //define x_0 -> x_15 as 1.

    int k_1 = 10;
    int k_2 = 16;
    int amount = 2000;

    std::vector<double> ans = generator(amount, k_1, k_2);
    //for(double b : ans)
    //{
    //    std::cout<<std::fixed<<std::setprecision(10)<<b;
    //} //now have numbers normalzied

    int value_t = 4;
    double j = 0.75;
    //probability is p = J.

    std::map<int, int> paramore = gaptest(ans, value_t, j);
    int sum = 0;

    for(auto const& pair : paramore)
    {
        //std::cout<<pair.first<<" "<<pair.second<<", ";
        sum += pair.second;
    }

    int firstmiss = 2000-sum;
    //std::cout<<firstmiss;

    paramore[1] = firstmiss;

    for(auto const& pair : paramore)
    {
        std::cout<<pair.first<<" "<<pair.second<<",";
        std::cout<<std::endl;
    }

    return 0;
}

std::vector<double> generator(int amount, int k_1, int k_2)
{
    std::vector<long long> ans = {};
    std::vector<double> steve = {};
    int start = std::max(k_1,k_2);

    for(int i = 0; i<start; i++)
        ans.push_back(1);

    int x_0 = ans[k_1];
    int x_1 = ans[k_2];
    int initial_ans_size = ans.size();

    for(int i = start; i<amount; i++)
    {
        long long given = ans[i - k_1] + ans[i - k_2];
        given = given % (1LL << 35);
        ans.push_back(given);
    }

    for(int i = 0; i<ans.size(); i++)
    {
        steve.push_back(ans[i] / static_cast<double>((1LL << 35)));
    }

    return steve;
}


std::map<int, int> gaptest(std::vector<double> ans, int val, double j)
{
    int counter = 0;
    std::map<int, int> epsilon;
    int t_more = 0;

    for(int i = 0; i < ans.size(); i++)
    {
        if(ans[i] < j) // End of a gap
        {
            if (counter > 0) // Only register non-zero gaps
            {
                if(counter > val)
                    t_more++;  // Counts gaps longer than `val`
                else
                    epsilon[counter]++;
            }
            counter = 0; // Reset for the next gap
        }
        else // ans[i] > j, so increase gap length
        {
            counter++;
        }
    }

    // Ensure the last gap is counted properly
    if(counter > val) 
        t_more++;
    else if(counter > 0)  // Make sure nonzero gaps are included
        epsilon[counter]++;

    epsilon[val+1] = t_more;

    return epsilon;
}
