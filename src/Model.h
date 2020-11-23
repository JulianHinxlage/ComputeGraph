//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_MODEL_H
#define COMPUTEGRAPH_MODEL_H

#include "Sequence.h"
#include "Optimizer.h"
#include "Loss.h"

class Model {
public:
    std::shared_ptr<Optimizer> optimizer;
    std::shared_ptr<Loss> loss;
    Sequence forward;
    Sequence backward;

    std::vector<Tensor> bestParameter;
    double bestLoss;

    Model();
    Model(Node &node);
    void compile(Node &node);
    const Tensor &predict(const Tensor &input);

    double samples(const Tensor &input, const Tensor &target, int samples = 1);
    double columnSamples(const Tensor &input, const Tensor &target, int epochs = 1);

    int totalParameterCount();
    void resetToBest();
    void checkBest(double lossValue);
};


#endif //COMPUTEGRAPH_MODEL_H
