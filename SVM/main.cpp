/*******************************************************************
 *  Copyright(c) 2016
 *  All rights reserved.
 *
 *  Name: Support Vector Machine
 *  Lib: \
 *  Date: 2016-3-6
 *  Author: Yang
 ******************************************************************/

#include <iostream>
#include <fstream>
#include "SVM.hpp"
#include <string>
#include <regex>
#include <algorithm>
#include <cctype>

void split(std::vector<std::string>& elements, std::string str, const std::string& regex)
{
    std::regex re(regex);
    std::sregex_token_iterator first{str.begin(), str.end(), re, -1}, last;
    
    std::move(first, last, std::back_inserter(elements));
}



int main()
{
    std::vector<std::vector<double>> data;
    std::vector<double> label;
    
    
    std::fstream fin("/Users/liuyang/Desktop/testSet.txt", std::ios::in);
    for (std::string line; std::getline(fin, line);)
    {
        std::vector<std::string> strTmp;
        split(strTmp, line, "\t");
        data.emplace_back(std::vector<double>{std::stod(strTmp[0]),
            std::stod(strTmp[1])});
        label.emplace_back(std::stod(strTmp[2]));
    }
    
    std::vector<std::vector<double>> train_data;
    std::vector<std::vector<double>> test_data;
    
    std::vector<double> train_label;
    std::vector<double> test_label;
    
    std::copy(data.begin(), data.begin() + 79, std::back_inserter(train_data));
    std::copy(label.begin(), label.begin() + 79, std::back_inserter(train_label));
    std::copy(data.begin() + 80, data.end(), std::back_inserter(test_data));
    std::copy(label.begin() + 80, label.end(), std::back_inserter(test_label));
    
    SupportVectorMachine svm(data, label, 0.6, kernel::LinearKernel);
    svm.train(3);
    
//    for (int i = 0; i < test_label.size(); i++)
//    {
//        std::cout << svm.predict(test_data[i]) << ' ' << test_label[i] << std::endl;
//    }
    
    std::vector<double> Omega = svm.get_omega();
    double b = svm.get_b();
    
    std::fstream fout("/Users/liuyang/Documents/MATLAB/omega_b.txt",
                      std::ios::out);
    
    std::for_each(Omega.begin(), Omega.end(),
                  [&fout](const double& n){  fout << n << std::endl;});
    fout << b;
}

