#ifndef SVM_hpp
#define SVM_hpp

#define EQUAL(a, b) assert(a == b);
#define UNEQUAL(a, b) assert(a == b);

#include <eigen3/Eigen/Eigen>

#include <vector>
#include <random>
#include <cmath>
#include <functional>
#include <stdexcept>

#include <iostream>

using namespace std::placeholders;
using Eigen::VectorXf;
using Eigen::MatrixXf;

namespace kernel
{
    double LinearKernel(const std::vector<double>&,
                        const std::vector<double>&);
    
    double GaussianKernel(const std::vector<double>&,
                          const std::vector<double>&, double);
}

class SupportVectorMachine
{
    using vectorD = std::vector<double>;
    using index_res = std::pair<int, double>;

    friend double kernel::LinearKernel(const std::vector<double>&,
                                       const std::vector<double>&);
    friend double kernel::GaussianKernel(const std::vector<double>&,
                                         const std::vector<double>&,
                                         double);
    
private:
    
    double C;
    double b;
    const double epsilon;
    vectorD alpha;
    vectorD label;
    vectorD omega;
    std::vector<vectorD> data;
    std::function<double(vectorD, vectorD)> kernel;
    
    double sum(vectorD::iterator, vectorD::iterator, const vectorD&);
    void add_vector(const vectorD&, vectorD&, double factor = 1);
    bool update(const index_res&, const index_res&);
    
public:
    explicit SupportVectorMachine(std::vector<vectorD>, vectorD,
                                  double, std::function<double(vectorD,
                                                               vectorD)>);
    explicit SupportVectorMachine(std::vector<vectorD> data,                                    vectorD label, vectorD alpha,                                double C, std::function<double(vectorD, vectorD)> kernel)
    : C(C), data(data), label(label), alpha(alpha),
    epsilon(0.001), kernel(kernel) {}

    void train(unsigned int);
    
    double predict(const vectorD&) const;
    
    std::vector<double> get_omega() const;
    double get_b() const;
};

#endif /* SVM_hpp */
