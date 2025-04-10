#include <vector>
#include <random>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

using namespace std;


vector<map<double, double>> pmfs()
{
    ifstream file("input.csv");
    string line;
    vector<double> space;
    vector<double> output;

    if (file.is_open())
    {
        if (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string cell;

            while (std::getline(ss, cell, ','))
            {
                space.push_back(std::stod(cell));
            }
        }

        if (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string cell;

            while (std::getline(ss, cell, ','))
            {
                output.push_back(std::stod(cell));
            }
        }

        file.close();
    }

    std::vector<std::map<double, double>> pmfs_;

    for(int i = 0; i < space.size(); i++)
    {
        std::map<double, double> temp;
        temp[space[i]] = output[i];
        pmfs_.push_back(temp);
        temp.clear();
    }

    return pmfs_;
}


vector<map<double,double>> cdf(vector<map<double,double>> epsilon)
{
    vector<map<double,double>> output;
    map<double, double> temp = {};
    double sum = 0;
    for(auto b : epsilon)
    {
        for(auto const& pair : b)
        {
            sum += pair.first;
            temp[sum] = pair.second;
        }
        output.push_back(temp);
    }

    return output;
}

map<int,int> generation(vector<map<double,double>> epsilon)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0,1);
    double random_num = dist(gen);

    int X = 0;

    map<int, int> output;
    bool flag;
    
    for(int i = 0; i<1000; i++)
    {
        for(auto b : epsilon)
        {
            for(auto const& pair : b)
            {
                if(random_num < pair.first)
                {
                    output[pair.second]++;
                    flag = true;
                }
            }
            if(flag == true)
            {
                flag = false;
                break;
            }
        }
        random_num = dist(gen);
    }

    return output;
}

int main()
{
    vector<map<double,double>> test = pmfs();
    vector<map<double,double>> cdfs = cdf(test);
    map<int, int> final = generation(cdfs);

    for(auto const& pair : final)
        cout<<pair.first<<", "<<pair.second<<endl;


    return 0;
}
