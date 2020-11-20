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

    Model();
    Model(Node &node);
    void compile(Node &node);
    double samples(const Tensor &input, const Tensor &target, int samples = 1);
    double columnSamples(const Tensor &input, const Tensor &target, int epochs = 1);
    int totalParameterCount();
};


#endif //COMPUTEGRAPH_MODEL_H
