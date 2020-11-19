//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_MODEL_H
#define COMPUTEGRAPH_MODEL_H

#include "Sequence.h"
#include "Adam.h"

class Model {
public:
    Adam optimizer;
    Sequence forward;
    Sequence backward;

    Model();
    Model(Node &node);
    void compile(Node &node);
    double sample(const Tensor &input, const Tensor &target);
    double columnSamples(const Tensor &input, const Tensor &target, int epochs = 1);
    double loss(const Tensor &output, const Tensor &target);
    Tensor lossGradient(const Tensor &output, const Tensor &target);
};


#endif //COMPUTEGRAPH_MODEL_H
