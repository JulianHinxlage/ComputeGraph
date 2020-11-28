//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_ACTORCRITICAGENT_H
#define COMPUTEGRAPH_ACTORCRITICAGENT_H

#include "Agent.h"
#include "model/Model.h"

class ActorCriticAgent : public Agent{
public:
    Model actorCritic;
    Model loss;

    ActorCriticAgent();
    void init(int inputs, int actions, const std::vector<int> &layers = {10}, const std::vector<int> &actorLayers = {10}, const std::vector<int> &criticLayers = {10}, double entropyFactor = 0.01);

    int sampleDistribution(const Tensor &probabilities);
    virtual int policyStep(const Tensor &state) override;
    virtual void trainStep(const Tensor &state, int action, double reward, double accumulativeReward, const Tensor &state2, int action2, bool isNextTerminal) override;
};


#endif //COMPUTEGRAPH_ACTORCRITICAGENT_H
