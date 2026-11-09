#pragma once

#include <Eigen/Dense>

Eigen::MatrixXf attention(const Eigen::MatrixXf &Q, const Eigen::MatrixXf &K,
                          const Eigen::MatrixXf &V);

struct MultiHeadAttention {
    int num_heads;
    int d_model;
    int d_k;
    Eigen::MatrixXf Wq_, Wk_, Wv_, Wo_;

    MultiHeadAttention(int num_heads, int d_model);
    Eigen::MatrixXf forward(const Eigen::MatrixXf &Q);
};
