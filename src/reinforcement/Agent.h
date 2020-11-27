//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_AGENT_H
#define COMPUTEGRAPH_AGENT_H

#include "graph/Tensor.h"

class Agent {
public:
    double discountFactor;

    class Step{
    public:
        Tensor state;
        int action;
        double reward;
        double accumulativeReward;
        bool terminal;
    };

    std::vector<Step> replayBuffer;
    int maxReplayBufferSize;
    bool updateAccumulativeRewards;
    bool onPolicyTrain;
    int timeStep;

    Agent();
    virtual int policyStep(const Tensor &state) = 0;
    int step(const Tensor &state, double reward, bool terminal);
    virtual void train(int steps);
    virtual void trainStep(const Tensor &state, int action, double reward, const Tensor &state2, int action2);
};


#endif //COMPUTEGRAPH_AGENT_H
