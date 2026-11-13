#include "feedforward.hpp"
#include "Eigen/src/Core/Matrix.h"
#include <Eigen/src/Core/VectorBlock.h>

// LayerNorm Function
// ignore gamma and beta
Eigen::MatrixXf LayerNorm
    (const Eigen::MatrixXf &X, const Eigen::MatrixXf &Z) {
    Eigen::MatrixXf add_mat = X + Z;
    auto Norm = [](const Eigen::MatrixXf& mat) -> Eigen::MatrixXf {
        Eigen::VectorXf mu = mat.rowwise().mean();
        Eigen::MatrixXf centered = mat.rowwise() - mu.transpose();
        Eigen::VectorXf var = centered.array().square().rowwise().mean();
        Eigen::MatrixXf norm = centered.array().rowwise() /
                                var.array().sqrt().transpose();

        //Eigen::MatrixXf out = norm.array().rowwise();
        return norm;
    };
    return Norm(add_mat);
}

Eigen::MatrixXf ReLU(const Eigen::MatrixXf& X) {
    Eigen::MatrixXf result = X;
    for (int i = 0; i < result.rows(); i++) {
        for (int j = 0; j < result.cols(); j++) {
            result(i,j) = std::max(0.0f, result(i, j));
        }
    }
    return result;
}

// FeedForward fully connected layer
// FFN(x) = ReLU(x * W1 +b1) * W2 + b2;
// ingore b1 and b2
Eigen::MatrixXf FeedForward(const Eigen::MatrixXf& x,
                            const Eigen::MatrixXf& W1,
                            const Eigen::MatrixXf& W2) {
    Eigen::MatrixXf hidden = x * W1;
    return ReLU(hidden) * W2;
}
