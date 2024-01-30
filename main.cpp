// TODO: Ask dad if: #include "Commmon.h"
#include "ProcessData.h"
#include "Computations.h"

int main() {
    // Set to the dataset path  
    const string FILE_PATH = "data/datasets/swissrole_10_0.csv";
    //const string FILE_PATH = "data/datasets/mnist_train_ext.csv";
    // Set to number of features in the dataset
    const int NUM_FEATURES = 3;
    // Set to desired K neighbors
    const int K_NEIGHBORS = 3;
    // Set to desired Polynomial Degree
    const int P_DEGREE = 3;
    
    // Import and process the data
    ProcessData pd(FILE_PATH, NUM_FEATURES);
    if (!pd.checkFile()) {
        cerr << "File failed to open \n";
        return -1;
    }
    pd.importData(); 
    pd.closeFile();

    // Run the NPPE algorithm on the data with K neighbors and P polynomial degree as inputs
    Computations c(pd.getData(), K_NEIGHBORS, P_DEGREE, NUM_FEATURES);
    c.runAlgorithm();
    
    return 0;
}

