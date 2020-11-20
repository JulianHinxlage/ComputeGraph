//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_MODEL_H
#define COMPUTEGRAPH_MODEL_H

#include "Sequence.h"
#include "Optimizer.h"

class Model {
public:
    std::shared_ptr<Optimizer> optimizer;
    Sequence forward;
    Sequence backward;

    Model();
    Model(Node &node);
    void compile(Node &node);
    double samples(const Tensor &input, const Tensor &target, int samples = 1);
    double columnSamples(const Tensor &input, const Tensor &target, int epochs = 1);
    double loss(const Tensor &output, const Tensor &target);
    Tensor lossGradient(const Tensor &output, const Tensor &target);
};


#endif //COMPUTEGRAPH_MODEL_H
