//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_OPTIMIZER_H
#define COMPUTEGRAPH_OPTIMIZER_H

#include "graph/Tensor.h"

class Optimizer {
public:
    typedef const std::function<void(const std::function<void(Tensor &, Tensor &)> &)> &Each;

    int batchSize;
    int sampleCounter;

    Optimizer(int batchSize = 1);
    virtual void update(Each each, int samples = 1);
    virtual void updateRule(Each each) = 0;
    static void reshape(Each each, std::vector<Tensor> &data);
};

class MomentumGradientDescent : public Optimizer{
public:
    double learningRate;
    double momentum;
    std::vector<Tensor> gradientMomentum;

    MomentumGradientDescent(double learningRate = 0.1, double momentum = 0.9, int batchSize = 1);
    virtual void updateRule(Each each) override;
};

class Adam : public Optimizer{
public:
    double learningRate;
    double beta1;
    double beta2;
    std::vector<Tensor> gradientMomentum1;
    std::vector<Tensor> gradientMomentum2;
    double beta1t;
    double beta2t;
    double decay;
    double epsilon;
    double gradientClip;

    Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, int batchSize = 1, double decay = 0, double epsilon = 1e-8, double gradientClip = std::numeric_limits<double>::max());
    virtual void updateRule(Each each) override;
};

#endif //COMPUTEGRAPH_OPTIMIZER_H
