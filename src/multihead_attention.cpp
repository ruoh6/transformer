#include "multihead_attention.hpp"
#include <cmath>

// scaled dot product attention
Eigen::MatrixXf attention(const Eigen::MatrixXf &Q,
                          const Eigen::MatrixXf &K,
                          const Eigen::MatrixXf &V) {
    Eigen::MatrixXf attn_score = Q * K.transpose() / std::sqrt(Q.cols());

    auto softmax = [](const Eigen::MatrixXf &mat) -> Eigen::MatrixXf {
        Eigen::MatrixXf row_max = mat.rowwise().maxCoeff();
        Eigen::MatrixXf max_broadcast = row_max.replicate(1, mat.cols());

        // avoid exp(attn_score) overflow
        Eigen::MatrixXf exp_mat = (mat - max_broadcast).array().exp();

        Eigen::MatrixXf row_sum = exp_mat.rowwise().sum();
        Eigen::MatrixXf sum_broadcast = row_sum.replicate(1, mat.cols());

        return exp_mat.array() / sum_broadcast.array();
    };

    return softmax(attn_score) * V;
}

MultiHeadAttention::MultiHeadAttention(int num_heads, int d_model) {
    this->num_heads = num_heads;
    this->d_model = d_model;
    this->d_k = d_model / num_heads;

    Wq_ = Eigen::MatrixXf::Random(d_model, d_model);
    Wk_ = Eigen::MatrixXf::Random(d_model, d_model);
    Wv_ = Eigen::MatrixXf::Random(d_model, d_model);
    Wo_ = Eigen::MatrixXf::Random(d_model, d_model);
}

Eigen::MatrixXf MultiHeadAttention::forward(const Eigen::MatrixXf &X) {

    // linear
    Eigen::MatrixXf Q = X * Wq_;
    Eigen::MatrixXf K = X * Wk_;
    Eigen::MatrixXf V = X * Wv_;

    // concat matrix
    Eigen::MatrixXf concat_mat(X.rows(), d_model);

    // multihead attention
    for (int h = 0; h < num_heads; h++) {
        // split head
        Eigen::MatrixXf Qh = Q.middleCols(h * d_k, d_k);
        Eigen::MatrixXf Kh = K.middleCols(h * d_k, d_k);
        Eigen::MatrixXf Vh = V.middleCols(h * d_k, d_k);

        // self attention
        Eigen::MatrixXf head = attention(Qh, Kh, Vh);
        concat_mat.middleCols(h * d_k, d_k) = head;
    }

    return concat_mat * Wo_;
}
