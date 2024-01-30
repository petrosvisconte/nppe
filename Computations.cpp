#include <algorithm>
#include <Eigen/Sparse>
#include <omp.h>
#include <mutex>
#include <iomanip> // precision
#include "Computations.h"

struct MatrixPair {
    Eigen::MatrixXd x;
    Eigen::VectorXd y;
};

// Constructor
Computations::Computations(const vector<vector<double>>& data_high_dim, int k_neighbors, int p_degree, int n_features){
    data_high_dim_ = data_high_dim;
    k_neighbors_ = k_neighbors;
    p_degree_ = p_degree;
    n_features_ = n_features;
} 

// Runs the main algorithm logic
vector<double> Computations::runAlgorithm() {
    cout << "Running training phase: \n" << std::setprecision(15);

    // Build a linear reconstruction weight matrix for all points in the dataset based on their K-nearest neighbors
    // Initialize a sparse matrix of size SxS (s=samples) to be used as the weight matrix
    Eigen::SparseMatrix<double, Eigen::RowMajor> linear_weight_matrix(data_high_dim_.size(), data_high_dim_.size());
    //linear_weight_matrix.reserve(k_neighbors_ * data_high_dim_.size());
    // Fill the linear weight matrix
    cout << "Computing linear reconstruction weights \n";
    //buildLinearWeights(linear_weight_matrix);
    cout << endl;
    
    // Build a non-linear reconstruction weight matrix for all points in the dataset based on the linear reconstruction weights
    // Initialize a sparse matrix of size SxS to be used as the weight matrix
    Eigen::SparseMatrix<double, Eigen::RowMajor> poly_weight_matrix(data_high_dim_.size(), data_high_dim_.size());
    //poly_weight_matrix.reserve(k_neighbors_ * data_high_dim_.size());
    // Fill the non-linear weight matrix
    cout << "Computing non-linear reconstruction weights \n";
    //buildPolyWeights(linear_weight_matrix, poly_weight_matrix);
    cout << endl;
    //cout << poly_weight_matrix;
    //cout << endl << endl;

    // Build a 3d matrix of Hadamard product from the P degree to the 1st degree for all points in the dataset
    // Initialize a 3D matrix of size SxPxN to be used as the weight matrix
    Eigen::Tensor<double, 3, Eigen::RowMajor> xp_3d_matrix;
    xp_3d_matrix.resize(data_high_dim_.size(), n_features_, p_degree_);
    // Fill the 3d Hadamard product matrix
    cout << "Generating X_p" << endl;
    buildXpMatrix(xp_3d_matrix);
    
    cout << xp_3d_matrix << endl << endl;

    return {0.0};
}

// Computes a matrix of Hadamard product from the P degree to the 1st degree for all points in the dataset
void::Computations::buildXpMatrix(Eigen::Tensor<double, 3, Eigen::RowMajor>& xp_3d_matrix) {
    std::mutex mtx;

    // Declare some variables for the progress bar
    int total_progress = data_high_dim_.size();
    int current_progress = 0;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    for (int i = 0; i < data_high_dim_.size(); i++) {
        int iter = 0;
        Eigen::Tensor<double, 2, Eigen::RowMajor> m;
        m.resize(p_degree_, n_features_);
        for (int p = p_degree_; p > 0; p--) {
            for (int j = 0; j < n_features_; j++) {
                double x_i = data_high_dim_[i][j];
                double x_i_initial = data_high_dim_[i][j];
                // Compute x_i to the p power
                for (int k = p-1; k > 0; k--) {
                    x_i = x_i * x_i_initial;
                }
                m(iter, j) = x_i;
            }
            iter++;
        }
        //cout << m << endl << endl;;
        xp_3d_matrix.chip(i, 0) = m;
        //iter++;

        mtx.lock();
        // Update the progress bar
        current_progress++;
        print_progress_bar(current_progress, total_progress, start_time);
        mtx.unlock();
    }

    // for (int i = 0; i < data_high_dim_.size(); i++) {
    //     int iter = 0;
    //     Eigen::Tensor<double, 2, Eigen::RowMajor> m;
    //     m.resize(p_degree_, n_features_);
    //     for (int p = p_degree_; p > 0; p--) {
    //         for (int j = 0; j < n_features_; j++) {
    //             double x_i = data_high_dim_[i][j];
    //             double x_i_initial = data_high_dim_[i][j];
    //             // Compute x_i to the p power
    //             for (int k = p-1; k > 0; k--) {
    //                 x_i = x_i * x_i_initial;
    //             }
    //             m(iter, j) = x_i;
    //         }
    //         iter++;
    //     }
    //     //cout << m << endl << endl;;
    //     xp_3d_matrix.chip(i, 0) = m;
    //     //iter++;

    //     mtx.lock();
    //     // Update the progress bar
    //     current_progress++;
    //     print_progress_bar(current_progress, total_progress, start_time);
    //     mtx.unlock();
    // }
}

// Computes the non-linear reconstruction weights for each data point from the linear reconstruction weights
void Computations::buildPolyWeights(Eigen::SparseMatrix<double, Eigen::RowMajor>& linear_weight_matrix, 
                                    Eigen::SparseMatrix<double, Eigen::RowMajor>& poly_weight_matrix) {
    std::mutex mtx;
    
    // Declare some variables for the progress bar
    int total_progress = data_high_dim_.size();
    int current_progress = 0;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    // Declare a vector of triplets to be used later to fill in the weight matrix
    vector<Eigen::Triplet<double>> weight_list;
    weight_list.reserve(k_neighbors_ * data_high_dim_.size());

    #pragma omp parallel for
    // Iterate over the non-zero entries of the linear weight matrix.
    // Note the outer for loops over the rows and the inner for loops over the rows (Due to the matrix being RowMajor)
    for (int i = 0; i < data_high_dim_.size(); i++) {
        // Declare a map to store the sums for each column index
        std::unordered_map<int, double> sums;
        // Iterate over the non-zero entries of row i
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(linear_weight_matrix, i); it; ++it) {
            // Get the column index and value of the current entry
            int j = it.col();
            double rij = it.value();
            // Iterate over the non-zero entries of row j
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it2(linear_weight_matrix, j); it2; ++it2) {
                // Get the column index and value of the current entry
                int k = it2.col();
                double rjk = it2.value();
                // Update the sum for column k
                sums[k] += rij * rjk;
            }
        }
        // Iterate over the non-zero entries of row i again
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(linear_weight_matrix, i); it; ++it) {
            // Get the column index and value of the current entry
            int j = it.col();
            double rij = it.value();
            // Calculate the nonlinear weight at index (i,j)
            // W(i,j) = R(i,j) + R(j,i) - Sum{R(i,k)*R(k,j)}
            double weight = rij + linear_weight_matrix.coeff(j,i) - sums[j];
            // Lock the mutex to avoid data races when pushing back the triplets to the weight list
            mtx.lock();
            // Push the weights for point i to the vector of triplets
            weight_list.push_back(Eigen::Triplet<double>(i, j, weight));
            mtx.unlock();
        }
        mtx.lock();
        // Update the progress bar
        current_progress++;
        print_progress_bar(current_progress, total_progress, start_time);
        mtx.unlock();
    }
    // Build the weight matrix from the vector of triplets
    poly_weight_matrix.setFromTriplets(weight_list.begin(), weight_list.end());

    // Normalize the weights
    
}

// Computes the linear reconstruction weights for each data point from its K-nearest neighbors
void Computations::buildLinearWeights(Eigen::SparseMatrix<double, Eigen::RowMajor>& weight_matrix) {
    std::mutex mtx;
    
    // Declare some variables for the progress bar
    int total_progress = data_high_dim_.size();
    int current_progress = 0;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    
    // Declare a vector of triplets to be used later to fill in the weight matrix
    vector<Eigen::Triplet<double>> weight_list;
    weight_list.reserve(k_neighbors_ * data_high_dim_.size());

    // Iterate over all points in the dataset
    #pragma omp parallel for
    for (int i = 0; i < data_high_dim_.size(); i++) {
        // Get the K-nearest neighbors data for point i as a K x N matrix and the neighbor indices as a K column-vector
        MatrixPair neighbor_pair = nearestNeighbors(data_high_dim_[i], i);
        Eigen::Matrix neighbor_matrix = neighbor_pair.x;
        Eigen::VectorXd neighbor_vector = neighbor_pair.y;
        
        // Initialize a K x N matrix where each row is the data for point i
        // Create a vector from the data for point i
        Eigen::VectorXd point_i = Eigen::Map<Eigen::VectorXd>(data_high_dim_[i].data(), n_features_);
        // Replicate the vector along the rows to create a K x N matrix
        Eigen::MatrixXd i_matrix = point_i.transpose().replicate(k_neighbors_, 1);

        // Subtract the point i matrix from the neighbors matrix to center the data about the origin
        neighbor_matrix.noalias() = neighbor_matrix - i_matrix;

        // Compute the local covariance
        neighbor_matrix = neighbor_matrix * neighbor_matrix.transpose().eval();
        // Regularize the matrix if the local covariance is not full rank (when K>N)
        if (k_neighbors_ > n_features_) {
            double e = 0.001 * neighbor_matrix.trace();
            neighbor_matrix = neighbor_matrix + Eigen::MatrixXd::Identity(k_neighbors_, k_neighbors_) * e;
        }

        // Solve the linear system for the weights
        Eigen::VectorXd col_vect_1 = Eigen::VectorXd::Ones(k_neighbors_);
        Eigen::VectorXd weights = neighbor_matrix.colPivHouseholderQr().solve(col_vect_1);

        // Normalize the weights
        double sum = weights.sum();
        weights = weights / sum;

        // Lock the mutex to avoid data races when pushing back the triplets to the weight list
        mtx.lock();
        // Push the weights for point i to the vector of triplets
        for (int j = 0; j < k_neighbors_; j++) {
            weight_list.push_back(Eigen::Triplet<double>(i, neighbor_vector(j), weights(j)));
        }
        // Update the progress bar
        current_progress++;
        print_progress_bar(current_progress, total_progress, start_time);
        mtx.unlock();
    }
    // Build the weight matrix from the vector of triplets
    weight_matrix.setFromTriplets(weight_list.begin(), weight_list.end());
}

// Finds the K-nearest neighbors for a given point and returns their data as a matrix
MatrixPair Computations::nearestNeighbors(const vector<double>& point, int point_index){
    // Declare a vector of vectors to store distances and the corresponding index
    vector<vector<double>> distances;
    distances.reserve(data_high_dim_.size());
    // Compute distance from input point to another point for all points in the dataset
    for (int i = 0; i < data_high_dim_.size(); i++) {
        // Ignore the input point
        if (i == point_index) {
            continue;
        }
        // Initialize a vector with the distance value: [0] and index of the other point: [1]
        vector<double> row = {distanceEuclidean(point, data_high_dim_[i]), static_cast<double>(i)};
        distances.push_back(row);
    }
    // Sort the vector of distances and indices by the distances
    sort(distances.begin(), distances.end(), compareDistances);
    // Initialize a matrix of size K x N as the data of the first K elements in the sorted vector of distances
    // Initialize a column-vector of size K storing the neighbor indices
    Eigen::MatrixXd neighbor_matrix(k_neighbors_, n_features_);
    Eigen::VectorXd neighbor_vector(k_neighbors_);
    for (int i = 0; i < k_neighbors_; i++) {
        neighbor_vector(i) = distances[i][1];
        for (int j = 0; j < n_features_; j++) {
            neighbor_matrix(i,j) = data_high_dim_[distances[i][1]][j];
        }
    }
    // Initialize a MatrixPair datastructure containing both the neighbor data and neighbor indices
    MatrixPair neighbor_pair = {neighbor_matrix, neighbor_vector};

    return neighbor_pair;
}

// Computes Euclidean distance between 2 points. 
// Note the result is left as a squared value to reduce computation time (the order of distances is preserved)
double Computations::distanceEuclidean(const vector<double>& point_1, const vector<double>& point_2) {
    double sum = 0;
    for (int i = 0; i < n_features_; i++) {
        sum += (point_1[i] - point_2[i]) * (point_1[i] - point_2[i]);
    }
    return sum;
}

// Compares the distance values of two points, used for sorting
bool compareDistances(const vector<double>& point_1, const vector<double>& point_2) {
    return point_1[0] < point_2[0];
}

// A function that prints the progress bar given the current and the total progress
// and the start time of the loop
void Computations::print_progress_bar(int current_progress, int total_progress, std::chrono::steady_clock::time_point start_time) {
    const int bar_length = 50;
    const char fill_symbol = '=';
    const char empty_symbol = ' ';
    const char start_symbol = '[';
    const char end_symbol = ']';
    const char* percentage_format = "%3d%%";
    const char* time_format = "%8.2fs  %8.2fs";
    int percentage = 100.0 * current_progress / total_progress;
    
    // Calculate the number of filled and empty sections
    int filled = bar_length * percentage / 100;
    int empty = bar_length - filled;
    
    // Build the progress bar string
    std::string bar;
    bar.push_back(start_symbol);
    bar.append(filled, fill_symbol);
    bar.append(empty, empty_symbol);
    bar.push_back(end_symbol);

    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - start_time);
    auto remaining_time = elapsed_time * (total_progress - current_progress) / current_progress;

    // Print the progress bar, the percentage, the elapsed time and the remaining time
    std::cout << "\r" << bar << " ";
    printf(percentage_format, percentage);
    std::cout << " ";
    printf(time_format, elapsed_time.count(), remaining_time.count());
    // Clear the output buffer
    std::cout << std::flush;
}
