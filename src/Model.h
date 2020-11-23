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
    std::shared_ptr<Loss> lossFunction;
    Sequence forward;
    Sequence backward;

    Model();
    Model(Node &node);
    void compile(Node &node);
    const Tensor &predict(const Tensor &input);
    const Tensor &gradient(const Tensor &gradient);
    void updateOptimizer(int samples);

    double fit(const Tensor &input, const Tensor &target, int samples = 1, int epochs = 1);
    double fitColumns(const Tensor &input, const Tensor &target, int epochs = 1);
    int totalParameterCount();
};


#endif //COMPUTEGRAPH_MODEL_H
