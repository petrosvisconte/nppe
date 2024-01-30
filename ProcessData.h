#ifndef NPPE_PROCESSDATA_H
#define NPPE_PROCESSDATA_H

#include "Common.h"

class ProcessData {
    public:
        ProcessData(string file_path, int num_features);
        void importData();
        bool checkFile();
        void closeFile();
        vector<vector<double>>& getData();
    private:
        vector<vector<double>> data_;
        string file_path_;
        ifstream file_;
        int num_features_;
};


#endif //NPPE_PROCESSDATA_H