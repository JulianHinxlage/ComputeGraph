//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "QAgent.h"

QAgent::QAgent(int actionCount) {
    this->actionCount = actionCount;
    explorationRate = 0.1;
    updateAccumulativeRewards = false;
    stepSize = 0.01;
}

double QAgent::getQ(const Tensor &state, int action) {
    return values.get(xt::concatenate(xt::xtuple(xt::flatten(state), (Tensor){(double)action})))(0);
}

void QAgent::setQ(const Tensor &state, int action, double q) {
    values.set(xt::concatenate(xt::xtuple(xt::flatten(state), (Tensor){(double)action})), q);
}

double QAgent::maxQ(const Tensor &state) {
    double max = -INFINITY;
    for(int i = 0; i < actionCount; i++){
        double q = getQ(state, i);
        if(q > max){
            max = q;
        }
    }
    return max;
}

int QAgent::argmaxQ(const Tensor &state) {
    double max = -INFINITY;
    int arg = 0;
    for(int i = 0; i < actionCount; i++){
        double q = getQ(state, i);
        if(q > max){
            max = q;
            arg = i;
        }
    }
    return arg;
}

int QAgent::policyStep(const Tensor &state) {
    //explore
    if(xt::random::rand<double>({1}, 0, 1)(0) < explorationRate){
        return xt::random::randint<int>({1}, 0, actionCount)(0);
    }

    //exploit
    return argmaxQ(state);
}

void QAgent::train(int steps) {
    if(replayBuffer.size() < steps){
        return;
    }

    for(int i = 0; i < steps; i++){
        int index = 0;
        do{
            index = xt::random::randint<int>({1}, 0, replayBuffer.size())(0);
        }while(replayBuffer[index].terminal);

        auto &state = replayBuffer[index].state;
        auto &action = replayBuffer[index].action;
        auto &reward = replayBuffer[index+1].reward;
        auto &state2 = replayBuffer[index+1].state;

        double delta = reward + discountFactor * maxQ(state2) - getQ(state, action);
        setQ(state, action, getQ(state, action) + stepSize * delta);
    }
}
