#include "ProcessData.h"
using std::stringstream;

// Constructor
ProcessData::ProcessData(string file_path, int num_features) {
    file_path_ = file_path;
    num_features_ = num_features;
}

// Imports data and saves to a vector of vectors
void ProcessData::importData() {
    cout << "Importing data: " << file_path_ << "\n";

    // Define the number of rows and columns
    const int rows = 600000;
    const int cols = num_features_;
    // Reserve the space for the data vector
    data_.reserve(rows);
    
    // Read each line of the file
    string line;
    while (getline(file_, line)) {
        stringstream ss(line);
        // Declare a vector to store the values in the line
        vector<double> row;
        row.reserve(cols);
        string value;
        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }
        // Push the row vector to the data vector
        data_.push_back(row);
    }
}

// Checks if file is open, returns false if not
bool ProcessData::checkFile() {
    file_.open(file_path_);
    return file_.is_open();
}

// Closes file
void ProcessData::closeFile() {
    file_.close();
}

// Get function 
vector<vector<double>>& ProcessData::getData() {
    return data_;
}