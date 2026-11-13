#include "feedforward.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <iostream>

int main() {
    Eigen::MatrixXf X = Eigen::MatrixXf::Random(3,3);
    std::cout << "print X" << "\n";
    std::cout  << X << std::endl;
    Eigen::MatrixXf Z = Eigen::MatrixXf::Random(3,3);
    std::cout << "print Z" << "\n";
    std::cout  << Z << std::endl;
    auto it = LayerNorm(X, Z);
    return 0;
}
