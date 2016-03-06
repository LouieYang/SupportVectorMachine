#include "SVM.hpp"

inline double SupportVectorMachine::sum(vectorD::iterator first, vectorD::iterator last, const vectorD& data)
{
    /**
     *  @Brief: return the sum of a vector
     */
    double res = 0;
    for (auto it = first; it != last; it++)
    {
        res += *it;
    }
    return res;
}

void SupportVectorMachine::add_vector(const vectorD &source,
                                      vectorD& dest, double factor)
{
    /**
     *  @Brief: Add two vector with factor multiplying
     *          dest += source * factor
     */
    
    EQUAL(source.size(), dest.size());
    for (unsigned long i = 0; i < dest.size(); ++i)
    {
        dest[i] += source[i] * factor;
    }
}

inline bool SupportVectorMachine::update(const index_res& alpha_i,
                                         const index_res& alpha_j)
{
    /**
     *  @Brief: Select two alphas and update their value
     *
     *  @Reference: Sequential Minimal Optimization
     *
     */
    
    if (alpha_i.first == alpha_j.first)
    {
        return false;
    }
    
    double L, H;
    double &__alpha_i = alpha[alpha_i.first];
    double &__alpha_j = alpha[alpha_j.first];
    
    double alpha_i_old = __alpha_i;
    double alpha_j_old = __alpha_j;
    
    double &__y_i = label[alpha_i.first];
    double &__y_j = label[alpha_j.first];
    auto &__x_i = data[alpha_i.first];
    auto &__x_j = data[alpha_j.first];
    
    
    if (__y_i == __y_j)
    {
        L = std::max(0., __alpha_i + __alpha_j - C);
        H = std::min(C, __alpha_i + __alpha_j);
    }
    else
    {
        L = std::max(0., __alpha_j - __alpha_i);
        H = std::min(C, C + __alpha_j - __alpha_i);
    }
    
    double eta = 2 * kernel(__x_i, __x_j) - kernel(__x_i, __x_i) - kernel(__x_j, __x_j);
    
    __alpha_j -= (__y_j * (alpha_i.second - alpha_j.second) / eta);
    
    if (__alpha_j > H)
    {
        __alpha_j = H;
    }
    else if (__alpha_j < L)
    {
        __alpha_j = L;
    }
    
    __alpha_i = __alpha_i + __y_i * __y_j * (alpha_j_old - __alpha_j);

    double b1 = b - alpha_i.second - __y_i * (__alpha_i - alpha_i_old)
    * kernel(__x_i, __x_i) - __y_j * (__alpha_j - alpha_j_old)
    * kernel(__x_i, __x_j);
    
    double b2 = b - alpha_j.second - __y_i * (__alpha_i - alpha_i_old)
    * kernel(__x_i, __x_j) - __y_j * (__alpha_j - alpha_j_old)
    * kernel(__x_j, __x_j);
    
    if (__alpha_i < C && __alpha_i > 0
        && __alpha_j > 0 && __alpha_j < C)
    {
        b = (b1 + b2) / 2.;
    }
    else if (__alpha_j < C && __alpha_j > 0)
    {
        b = b2;
    }
    else
    {
        b = b1;
    }
    
    return true;
}

SupportVectorMachine::SupportVectorMachine(std::vector<vectorD> data,
                        vectorD label,double C,
                        std::function<double(vectorD,vectorD)> kernel)
: C(C), data(data), label(label), b(0), epsilon(0.001), kernel(kernel)
{
    /**
     *  @Brief: Set the default alpha and b to be 0
     */
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0, std::min(C, 2./data.size()));
    if (data.size() != label.size() || data.size() * C < 1)
    {
        std::logic_error("Size unfitted\n");
    }
    
    alpha = std::vector<double>(label.size(), 0);
    omega = std::vector<double>(data[0].size(), 0);
}

void SupportVectorMachine::train(unsigned int maxIter)
{
    /**
     *  @Brief: Train the SVM
     *
     *  @Reference: Sequential Minimal Optimization
     *              http://blog.csdn.net/zouxy09/article/details/17292011
     */
    unsigned int iter = 0;
    auto between_0_C = [this](const double& a)
    {
        return a < C && a > 0;
    };

    
    while (iter < maxIter)
    {
        std::vector<index_res> index_res_v, index_error_v;
        std::vector<double> old_predict;
        
        /// Make the predict and the error of the input data in the cache
        for (unsigned i = 0; i < alpha.size(); ++i)
        {
            double _tmp = 0;
            for (unsigned j = 0; j < alpha.size(); ++j)
            {
                _tmp = _tmp +
                alpha[j] * label[j] * kernel(data[i], data[j]);
            }
            _tmp += b;
            old_predict.emplace_back(_tmp);
            index_error_v.emplace_back(std::make_pair(i,
                                                      _tmp - label[i]));
        }
        
        /// First select the alpha between 0 and C
        for (auto it = std::find_if(alpha.begin(), alpha.end(), between_0_C); it != alpha.end(); it = std::find_if(it, alpha.end(), between_0_C))
        {
            auto i = std::distance(alpha.begin(), it);
            index_res_v.emplace_back(std::make_pair(i, old_predict[i]));
            std::advance(it, 1);
        }
        
        
        auto first_alpha = index_error_v.begin();
        if (!index_res_v.empty())
        {
            /**
             *  If there is an alpha between the (0, C)
             */
            auto index_res_v_m1 = index_res_v;
            for (unsigned int i = 0; i < index_res_v_m1.size(); ++i)
            {
                index_res_v_m1[i].second = index_res_v_m1[i].second
                * label[i] - 1;
                index_res_v_m1[i].second = std::abs(index_res_v_m1[i].second);
            }
            /// Find the worst alpha that doesn't fit kkt
            auto bad_alpha = std::max_element(index_res_v_m1.begin(),
                                            index_res_v_m1.end(),
                                    [](const std::pair<int, double>& p1,
                                       const std::pair<int, double>& p2)
                                    {
                                        return p1.second < p2.second;
                                    });
            std::advance(first_alpha, bad_alpha->first);
        }
        else
        {
            /**
             *  Else try to go through the whole training set, and find
             *  the alpha that violate the kkt most.
             */
            unsigned index = 0;
            double unkkt_error = 0;
            for (unsigned i = 0; i < alpha.size(); ++i)
            {
                if (alpha[i] >= 0 && alpha[i] <= epsilon)
                {
                    double error = 1 - old_predict[i] * label[i];
                    if (error > unkkt_error)
                    {
                        index = i;
                        unkkt_error = error;
                    }
                }
                else if (alpha[i] <= C && alpha[i] >= C - epsilon)
                {
                    double error = old_predict[i] * label[i] - 1;
                    if (error > unkkt_error)
                    {
                        index = i;
                        unkkt_error = error;
                    }
                }
            }
            std::advance(first_alpha, index);
        }

        /// find the second alpha according to the first one selected.
        std::vector<index_res>::iterator second_alpha;
        if (first_alpha->second > 0)
        {
            second_alpha = std::min_element(index_error_v.begin(),
                                        index_error_v.end(),
                                        [](const index_res& a,
                                           const index_res& b)
                                        {
                                            return a.second < b.second;
                                        });
        }
        else
        {
            second_alpha = std::max_element(index_error_v.begin(),
                                        index_error_v.end(),
                                        [](const index_res& a,
                                           const index_res& b)
                                        {
                                            return a.second < b.second;
                                        });
        }

        if (!update(*first_alpha, *second_alpha))
        {
            /**
             *  If the first_alpha == second_alpha which means |E_i - E_j|
             *  = 0 (the SVM converged), break the while loop.
             */
            break;
        }
        
        iter++;
    }
    
    /**
     *  Calculate the omega so that could make predict easier.
     */
    for (unsigned long i = 0; i < data.size(); ++i)
    {
        add_vector(data[i], omega, label[i] * alpha[i]);
    }
}

double SupportVectorMachine::predict(const std::vector<double>& X) const
{
    /**
     *  The SVM predict function
     *
     *  @param X: Input data to be classified.
     *
     *  @return: the Sign of the result show the classification
     */
    EQUAL(X.size(), omega.size());
    double res = 0;
    for (unsigned long i = 0; i < X.size(); ++i)
    {
        res += X[i] * omega[i];
    }
    res += b;
    return res;
}

double kernel::LinearKernel(const std::vector<double> &W,
                            const std::vector<double> &X)
{
    /**
     *  Show the most simple and trivia kernel function
     */
    double _res = 0;
    EQUAL(W.size(), X.size());
    
    for (unsigned long i = 0; i < W.size(); ++i)
    {
        _res += W[i] * X[i];
    }
    return _res;
}

double kernel::GaussianKernel(const std::vector<double> &X,
                              const std::vector<double> &Y,
                              double sigma)
{
    /**
     *  RBF kernel function
     */
    double _res = 0;
    EQUAL(X.size(), Y.size());
    UNEQUAL(sigma, 0);
    
    for (unsigned long i = 0; i < X.size(); ++i)
    {
        double diff = X[i] - Y[i];
        _res += diff * diff;
    }
    _res = _res / (-sigma * sigma);
    _res = std::exp(_res);
    return _res;
}

std::vector<double> SupportVectorMachine::get_omega() const
{
    /**
     *  Get the omega to draw in MATLAB
     */
    return omega;
}

double SupportVectorMachine::get_b() const
{
    /**
     *  Get the b to draw in MATLAB
     */
    return b;
}