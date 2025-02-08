#include <iostream>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <random>
#include <stdint.h>
#include <math.h>
#include "Mersenne.h"


#define n 624
#define m 397
#define w 32
#define r 31
#define UMASK (0xffffffffUL << r)
#define LMASK (0xffffffffUL >> (w-r))
#define a 0x9908b0dfUL
#define u 11
#define s 7
#define t 15
#define l 18
#define b 0x9d2c5680UL
#define c 0xefc60000UL
#define f 1812433253UL

using namespace std;

typedef struct 
{
    uint32_t state_array[n];
    int state_index; //where 0 <= state_index <= n-1
} mt_state;

void initialize_state(mt_state* state, uint32_t seed)
{
    uint32_t* state_array = &(state->state_array[0]);

    state_array[0] = seed;

    for(int i=1; i<n; i++)
    {
        seed = f * (seed ^ (seed >> (w-2))) + i;
        state_array[i] = seed;
    }

    state -> state_index = 0;
}

uint32_t random_unit32(mt_state* state)
{
    uint32_t* state_array = &(state->state_array[0]);
    int k = state->state_index;

    int j = k - (n-1);               // point to state n-1 iterations before
    if (j < 0) j += n;               // modulo n circular indexing

    uint32_t x = (state_array[k] & UMASK) | (state_array[j] & LMASK);
    
    uint32_t xA = x >> 1;
    if (x & 0x00000001UL) xA ^= a;
    
    j = k - (n-m);                   // point to state n-m iterations before
    if (j < 0) j += n;               // modulo n circular indexing
    
    x = state_array[j] ^ xA;         // compute next value in the state
    state_array[k++] = x;            // update new state value
    
    if (k >= n) k = 0;               // modulo n circular indexing
    state->state_index = k;
    
    uint32_t y = x ^ (x >> u);       // tempering 
             y = y ^ ((y << s) & b);
             y = y ^ ((y << t) & c);
    uint32_t z = y ^ (y >> l);

    return z;
}

std::vector<double> generateMersenne(uint32_t seed, int amount)
{
    mt_state state;
    initialize_state(&state, seed);

    vector<double> nums(amount);
    for(int i = 0; i<amount; i++)
    {
        uint32_t ranval = random_unit32(&state);
        //nums[i] = ranval;
        nums[i] = static_cast<double>(ranval) / static_cast<double>(UINT32_MAX);
        //nums[i] = nums[i] / pow(2,29);
    }

    return nums;
}

//int main()
//{
//    mt_state state;
//    uint32_t seed = 123456781;
//
//    initialize_state(&state, seed);
//
//    vector<int> nums(100);
//    for(int i = 0; i<100; i++)
//        nums[i] = random_unit32(&state);
//
//    for(int i = 0; i<10; i++)
//    {
//        nums[i] =  nums[i] / pow(2,29);
//        cout<<nums[i]<<" ";
//    }
//
//}




//generalized feedback shift register generators
//Mersenne is a GFSR
// << is the bitwise right shift operator
//MT19937 variant, begins with initial state which is a large array of integers. Uses array of 624 integers
// UMASK and LMASK isolates the most significant digit and least signifcant respectively 




