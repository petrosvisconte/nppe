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
    // for (int i = 0; i < data_high_dim_.size(); i++) {
    //     for (int j = 0; j < n_features_; j++) {
    //         cout << data_high_dim_[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    cout << "Running training phase: \n" << std::setprecision(15);

    // Build a linear reconstruction weight matrix for all points in the dataset based on their K-nearest neighbors
    // Initialize a sparse matrix of size SxS (s=samples) to be used as the weight matrix
    Eigen::SparseMatrix<double, Eigen::RowMajor> linear_weight_matrix(data_high_dim_.size(), data_high_dim_.size());
    linear_weight_matrix.reserve(k_neighbors_ * data_high_dim_.size());
    // Fill the linear weight matrix
    cout << "Computing linear reconstruction weights \n";
    buildLinearWeights(linear_weight_matrix);
    cout << endl;
    //cout << linear_weight_matrix << endl;
    
    // Build a non-linear reconstruction weight matrix for all points in the dataset based on the linear reconstruction weights
    // Initialize a sparse matrix of size SxS to be used as the weight matrix
    Eigen::SparseMatrix<double, Eigen::RowMajor> poly_weight_matrix(data_high_dim_.size(), data_high_dim_.size());
    poly_weight_matrix.reserve(k_neighbors_ * data_high_dim_.size());
    // Fill the non-linear weight matrix
    cout << "Computing non-linear reconstruction weights \n";
    buildPolyWeights(linear_weight_matrix, poly_weight_matrix);
    cout << endl;
    //cout << poly_weight_matrix;

    // Build a 2d matrix of Hadamard products from the P degree to the 1st degree for all points in the dataset
    // Initialize a 2D matrix of size P*NxS to be used as the X_p matrix
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xp_3d_matrix(data_high_dim_.size(), n_features_*p_degree_);
    // Fill the 3d Hadamard product matrix
    cout << "Generating X_p" << endl;
    buildXpMatrix(xp_3d_matrix);
    cout << endl;

    
    // Solve the generalized eigenvalue problem to obtain the eigenvectors for the m smallest eigenvalues 
    cout << "Solving generalized eigenvalue problem \n";
    eigvecs_ = solveEigenProblem(poly_weight_matrix, xp_3d_matrix);
    
    // Map each point in the dataset to its low dimensional representation and write output to file
    cout << "Mapping data to low dimensional space" << endl;
    mapLowDimension(xp_3d_matrix);

    // Map each point in the testing set to its low dimensional representation and write output to file
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> phi_matrix_test(10, n_features_*p_degree_);
    //buildXpMatrix(phi_matrix_test);
    //mapLowDimension(eigvecs, phi_matrix_test);


    return {0.0};
}

void Computations::mapLowDimensionTest(const vector<vector<double>>& data_high_dim_test) {    
    cout << "Mapping test data to low dimensional space" << endl;
    // Create a file to write the low dimensional data to
    string filename = "data/datasets/low_dim_test.csv";
    std::ofstream file(filename);
    if (file.fail()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> phi_matrix(data_high_dim_test.size(), n_features_*p_degree_);
    for (int i = 0; i < data_high_dim_test.size(); i++) {
        int iter = 0;
        for (int p = p_degree_; p > 0; p--) {
            for (int j = 0; j < n_features_; j++) {
                double x_i = data_high_dim_test[i][j];
                double x_i_initial = data_high_dim_test[i][j];
                // Compute x_i to the p power
                for (int k = p-1; k > 0; k--) {
                    x_i = x_i * x_i_initial;
                }
                phi_matrix(i, iter) = x_i;
                iter++;
            }
        }
    }
    // Write the low dimensional data to the file
    for (int i = 0; i < data_high_dim_test.size(); i++) {
        Eigen::VectorXd point(n_features_);
        for (int j = 0; j < 2; j++) { // TODO: update j < 2 to be a variable j < m, with m representing low dimensional space
            //cout << eigvecs.row(j) << "  :  \n" << xp_3d_matrix.row(i).transpose() << endl;
            point(j) = (eigvecs_.row(j) * phi_matrix.row(i).transpose())(0);
            file << point(j) << ",";
            //cout << endl << "point: " << point(j) << endl;
        }
        file << std::endl;
    }
    
    // close the file
    file.close();
}

void Computations::mapLowDimension(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& xp_3d_matrix) {    
    // Create a file to write the low dimensional data to
    string filename = "data/datasets/low_dim.csv";
    std::ofstream file(filename);
    if (file.fail()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    // Write the low dimensional data to the file
    for (int i = 0; i < data_high_dim_.size(); i++) {
        Eigen::VectorXd point(n_features_);
        for (int j = 0; j < 2; j++) { // TODO: update j < 2 to be a variable j < m, with m representing low dimensional space
            //cout << eigvecs.row(j) << "  :  \n" << xp_3d_matrix.row(i).transpose() << endl;
            point(j) = (eigvecs_.row(j) * xp_3d_matrix.row(i).transpose())(0);
            file << point(j) << ",";
            //cout << endl << "point: " << point(j) << endl;
        }
        file << std::endl;
    }
    
    // close the file
    file.close();
}

Eigen::MatrixXd Computations::solveEigenProblem(Eigen::SparseMatrix<double, Eigen::RowMajor>& poly_weight_matrix, 
                                      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& xp_3d_matrix) {
    // Create matrix D
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(data_high_dim_.size(), data_high_dim_.size());
    
    // Create matrix A and B for the generalized eigenvalue problem
    Eigen::MatrixXd A = xp_3d_matrix.transpose() * (D - poly_weight_matrix.transpose()) * xp_3d_matrix;
    Eigen::MatrixXd B = xp_3d_matrix.transpose() * D * xp_3d_matrix;

    // Solve the generalized eigenvalue problem 
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(A, B);
    Eigen::VectorXd eigvals = solver.eigenvalues(); // Get the eigenvalues as a vector
    Eigen::MatrixXd eigvecs = solver.eigenvectors(); // Get the eigenvectors as a matrix

    //cout << A << endl << endl;
    //cout << std::boolalpha << A.isApprox(A.adjoint()) << endl << endl;
    //cout << B << endl << endl;
    //cout << std::boolalpha << B.isApprox(B.adjoint()) << endl << endl;
    //cout << B.eigenvalues() <<  endl << endl;
    //cout << eigvals << endl << endl;
    //cout << eigvecs << endl << endl;

    //cout << A * eigvecs.row(0).transpose() << endl << endl;
    //cout << eigvals.coeff(0) * B * eigvecs.row(0).transpose() << endl << endl;;

    // Find the m smallest eigenvalues that satisfy the constraint
    //Eigen::VectorXd first_row = eigvecs.row(1);

    //cout << first_row << endl << endl;
    //cout << xp_3d_matrix.transpose() << endl << endl;
    //cout << D << endl << endl;
    //cout << xp_3d_matrix << endl << endl;
    //cout << first_row.transpose() << endl << endl;

    for (int i = 0; i < eigvals.size(); i++) {
        Eigen::Vector<double, Eigen::Dynamic> first_row = eigvecs.row(i);
        for (int j = 0; j < eigvals.size(); j++) {
            Eigen::Vector<double, Eigen::Dynamic> row = eigvecs.row(j);
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> constraint = 
                first_row.transpose() * xp_3d_matrix.transpose() * D * xp_3d_matrix * row;
            //cout << constraint << ", ";
        }
        //cout << endl;
    }
    //cout << endl;
    //cout << endl << eigvals.coeff(0) << endl;
    return eigvecs;
}

// Computes a matrix of Hadamard product from the P degree to the 1st degree for all points in the dataset
void Computations::buildXpMatrix(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& xp_3d_matrix) {
    std::mutex mtx;

    // Declare some variables for the progress bar
    int total_progress = data_high_dim_.size();
    int current_progress = 0;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    for (int i = 0; i < data_high_dim_.size(); i++) {
        int iter = 0;
        for (int p = p_degree_; p > 0; p--) {
            for (int j = 0; j < n_features_; j++) {
                double x_i = data_high_dim_[i][j];
                double x_i_initial = data_high_dim_[i][j];
                // Compute x_i to the p power
                for (int k = p-1; k > 0; k--) {
                    x_i = x_i * x_i_initial;
                }
                xp_3d_matrix(i, iter) = x_i;
                iter++;
            }
        }
        mtx.lock();
        // Update the progress bar
        current_progress++;
        print_progress_bar(current_progress, total_progress, start_time);
        mtx.unlock();
    }
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
    // Note the outer for loops over the rows and the inner for loops over the columns (Due to the matrix being RowMajor)
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


    //Normalize the weights TODO: decide to normalize or not
    //Get the sum of each row
    // Eigen::Vector<double, Eigen::Dynamic> rowsum(data_high_dim_.size());
    // #pragma omp parallel for
    // for (int i = 0; i < poly_weight_matrix.rows(); i++) {
    //     rowsum.coeffRef(i) = poly_weight_matrix.innerVector(i).sum();
    // }
    // // Divide each nonzero element by the sum of its corresponding row
    // #pragma omp parallel for
    // for (int i = 0; i < poly_weight_matrix.rows(); i++) {
    //     for (int j = 0; j < poly_weight_matrix.cols(); j++) {
    //         mtx.lock();
    //         if (poly_weight_matrix.coeffRef(i, j) != 0.0) {
    //             poly_weight_matrix.coeffRef(i,j) = poly_weight_matrix.coeffRef(i,j) / rowsum.coeffRef(i);
    //         }
    //         mtx.unlock();
    //     }
    // }
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
        MatrixPair neighbor_pair = nearestNeighbors(data_high_dim_[i], i); //TODO: return by reference
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> neighbor_matrix = neighbor_pair.x;
        Eigen::Vector<double, Eigen::Dynamic> neighbor_vector = neighbor_pair.y;
        //cout << neighbor_matrix << endl;
        //cout << neighbor_vector << endl;
        
        // Initialize a K x N matrix where each row is the data for point i
        // Create a vector from the data for point i
        Eigen::VectorXd point_i = Eigen::Map<Eigen::VectorXd>(data_high_dim_[i].data(), n_features_);
        // Replicate the vector along the rows to create a K x N matrix
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> i_matrix = point_i.transpose().replicate(k_neighbors_, 1);

        // Subtract the point i matrix from the neighbors matrix to center the data about the origin
        //neighbor_matrix = neighbor_matrix - i_matrix;

        // Compute the local covariance
        neighbor_matrix = neighbor_matrix * neighbor_matrix.transpose().eval();
        // Regularize the matrix if the local covariance is not full rank (when K>N)
        if (k_neighbors_ > n_features_) {
            double e = 0.001 * neighbor_matrix.trace();
            neighbor_matrix = neighbor_matrix + 
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(k_neighbors_, k_neighbors_) * e;
        }

        // Solve the linear system for the weights
        Eigen::VectorXd col_vect_1 = Eigen::VectorXd::Ones(k_neighbors_);
        Eigen::VectorXd weights = neighbor_matrix.colPivHouseholderQr().solve(col_vect_1);
        // Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver;
        // solver.compute(neighbor_matrix.transpose() * neighbor_matrix);
        // Eigen::VectorXd weights = solver.solve(col_vect_1);
        // Eigen::VectorXd weights = neighbor_matrix.partialPivLu().solve(col_vect_1);

        // Normalize the weights TODO: decide to normalize or not
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
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> neighbor_matrix(k_neighbors_, n_features_);
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
