//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_MEANBUFFER_H
#define COMPUTEGRAPH_MEANBUFFER_H

#include <vector>

class MeanBuffer {
public:
    std::vector<double> values;
    int index;
    int maxValues;
    double sum;

    MeanBuffer(int maxValues = 100);
    void add(double value);
    double mean();
    void clear();
};


#endif //COMPUTEGRAPH_MEANBUFFER_H
