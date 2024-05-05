#ifndef NPPE_COMPUTATIONS_H
#define NPPE_COMPUTATIONS_H

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/CXX11/Tensor>
#include <chrono>
#include "Common.h"

struct MatrixPair;

class Computations {
    public:
        Computations(const vector<vector<double>>& data_high_dim, int k_neighbors, int p_degree, int n_features);
        vector<double> runAlgorithm();
        void publicMapLowDimensionTest(const vector<vector<double>>& data_high_dim_test) {
            mapLowDimensionTest(data_high_dim_test);
        }
    private:
        void mapLowDimensionTest(const vector<vector<double>>& data_high_dim_test);
        void mapLowDimension(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& xp_3d_matrix);
        Eigen::MatrixXd solveEigenProblem(Eigen::SparseMatrix<double, Eigen::RowMajor>& poly_weight_matrix, 
                               Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& xp_3d_matrix);
        void buildXpMatrix(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& xp_3d_matrix);
        //void buildXpMatrix(Eigen::Tensor<double, 3, Eigen::RowMajor>& xp_3d_matrix);
        void buildPolyWeights(Eigen::SparseMatrix<double, Eigen::RowMajor>& linear_weight_matrix, 
                              Eigen::SparseMatrix<double, Eigen::RowMajor>& poly_weight_matrixx);
        void buildLinearWeights(Eigen::SparseMatrix<double, Eigen::RowMajor>& weight_matrix);
        MatrixPair nearestNeighbors(const vector<double>& point, int point_index);
        double distanceEuclidean(const vector<double>& point_1, const vector<double>& point_2);
        void print_progress_bar(int current_progress, int total_progress, std::chrono::steady_clock::time_point start_time);
        vector<vector<double>> data_high_dim_;
        vector<vector<double>> data_low_dim_;
        int k_neighbors_;
        int p_degree_;
        int n_features_;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigvecs_;
};
bool compareDistances(const vector<double>& point_1, const vector<double>& point_2);


#endif //NPPE_COMPUTATIONS_H