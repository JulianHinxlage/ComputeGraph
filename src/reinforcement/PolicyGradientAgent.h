//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_POLICYGRADIENTAGENT_H
#define COMPUTEGRAPH_POLICYGRADIENTAGENT_H

#include "Agent.h"
#include "model/Model.h"
#include "MeanBuffer.h"

class PolicyGradientAgent : public Agent {
public:
    Model policy;
    Model loss;
    MeanBuffer baseline;
    MeanBuffer variance;

    PolicyGradientAgent();
    void init(int inputs, int actions, const std::vector<int> &layers);

    int sampleDistribution(const Tensor &probabilities);
    virtual int policyStep(const Tensor &state) override;
    virtual void trainStep(const Tensor &state, int action, double reward, double accumulativeReward, const Tensor &state2, int action2) override;
};


#endif //COMPUTEGRAPH_POLICYGRADIENTAGENT_H
