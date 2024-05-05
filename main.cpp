// TODO: Ask dad if: #include "Commmon.h"
#include "ProcessData.h"
#include "Computations.h"
#include <cstdlib>

int main() {
    // Set to the dataset path  
    const string FILE_PATH = "data/datasets/swissroleh_2000_0.csv";
    //const string FILE_PATH = "data/datasets/mnist_train_ext.csv";
    // Set to number of features in the dataset (high dimensional space)
    const int NUM_FEATURES = 3;
    // Set to desired K neighbors
    const int K_NEIGHBORS = 5;
    // Set to desired Polynomial Degree
    const int P_DEGREE = 2;
    
    // Import and process the data
    ProcessData pd(FILE_PATH, NUM_FEATURES);
    if (!pd.checkFile()) {
        cerr << "File failed to open \n";
        return -1;
    }
    pd.importData(); 
    pd.closeFile();

    // Run the NPPE algorithm on the data with K neighbors and P polynomial degree as arguments
    Computations c(pd.getData(), K_NEIGHBORS, P_DEGREE, NUM_FEATURES);
    c.runAlgorithm();

    // Import and process the test data
    ProcessData pd_test("data/datasets/swissroleh_test.csv", NUM_FEATURES);
    if (!pd_test.checkFile()) {
        cerr << "File failed to open \n";
        return -1;
    }
    pd_test.importData(); 
    pd_test.closeFile();
    
    // Map the test data to the low dimensional space
    c.publicMapLowDimensionTest(pd_test.getData());
    
    return 0;
}

