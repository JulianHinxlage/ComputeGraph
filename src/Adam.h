//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_ADAM_H
#define COMPUTEGRAPH_ADAM_H

#include "Operations.h"

class Adam {
public:
    typedef const std::function<void(const std::function<void(Tensor &, Tensor &)> &)> &Each;

    double learningRate;
    int batchSize;
    int sampleCounter;
    double beta1;
    double beta2;
    std::vector<Tensor> gradientMomentum1;
    std::vector<Tensor> gradientMomentum2;
    double beta1t;
    double beta2t;
    double parameterDecayRate;

    Adam(double learningRate = 0.01, double beta1 = 0.9, double beta2 = 0.999, int batchSize = 1);
    void update(Each each, int samples = 1);
    void updateRule(Each each);
};


#endif //COMPUTEGRAPH_ADAM_H
