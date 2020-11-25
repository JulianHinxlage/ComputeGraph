//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_LOSS_H
#define COMPUTEGRAPH_LOSS_H

#include "graph/Tensor.h"

class Loss {
public:
    virtual double value(const Tensor &output, const Tensor &target) = 0;
    virtual Tensor gradient(const Tensor &output, const Tensor &target) = 0;
};

class MeanSquaredError : public Loss{
public:
    virtual double value(const Tensor &output, const Tensor &target);
    virtual Tensor gradient(const Tensor &output, const Tensor &target);
};

class CrossEntropy : public Loss{
public:
    virtual double value(const Tensor &output, const Tensor &target);
    virtual Tensor gradient(const Tensor &output, const Tensor &target);
};

class BinaryCrossEntropy : public Loss{
public:
    virtual double value(const Tensor &output, const Tensor &target);
    virtual Tensor gradient(const Tensor &output, const Tensor &target);
};

#endif //COMPUTEGRAPH_LOSS_H
