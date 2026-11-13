#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <cmath>
#include "multihead_attention.hpp"

using Eigen::MatrixXd;
using Eigen::MatrixXf;

TEST(EigenTest, MatrixElementAssignment) {
    MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    EXPECT_DOUBLE_EQ(m(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 2.5);
    EXPECT_DOUBLE_EQ(m(0, 1), -1.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 1.5);
}

TEST(AttentionTest, ManualCalculation) {
    MatrixXf Q(2, 2);
    Q << 1, 0, 0, 1;
    MatrixXf K = Q;
    MatrixXf V(2, 3);
    V << 2, 4, 6, 6, 8, 10;

    const float weight = std::exp(1.0f / std::sqrt(2.0f));
    const float p = weight / (weight + 1.0f);
    MatrixXf expected(2, 3);
    expected.row(0) = p * V.row(0) + (1.0f - p) * V.row(1);
    expected.row(1) = (1.0f - p) * V.row(0) + p * V.row(1);

    EXPECT_TRUE(attention(Q, K, V).isApprox(expected, 1e-5f));
}

TEST(AttentionTest, MultiHeadForward) {
    MultiHeadAttention layer(2, 4);
    layer.Wq_.setZero();
    layer.Wk_.setZero();
    layer.Wv_.setIdentity();
    layer.Wo_.setIdentity();

    MatrixXf X(3, 4);
    X << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
    MatrixXf expected(3, 4);
    expected << 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8;

    MatrixXf output = layer.forward(X);
    ASSERT_EQ(output.rows(), X.rows());
    ASSERT_EQ(output.cols(), X.cols());
    EXPECT_TRUE(output.isApprox(expected, 1e-5f));
}
