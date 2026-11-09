#include <Eigen/Dense>
#include <gtest/gtest.h>

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

TEST(AttentionTest, ManualCalculation) {}
