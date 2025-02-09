#include <iostream>
#include <bit>
#include <vector>
#include <cmath>
#include <stdint.h>
#include <math.h>
#include <algorithm>
#include <bitset>
#include <iomanip>

using namespace std;

int main()
{
    int N = 30;

    double ans;
    double mult = 1;

    for(int i = 0; i<N; i++)
    {
        if(i >= 2)
        {
            for(int j = i; j > 1; j--)
            {
                mult *= j;
            }
        }
        ans += 1/mult;
        mult = 1;
    }

    //ans += 1;
    cout<<fixed<<setprecision(10)<<ans;



    return 0;
}
