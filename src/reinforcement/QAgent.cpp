//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "QAgent.h"

QAgent::QAgent(int actionCount) {
    this->actionCount = actionCount;
    updateAccumulativeRewards = false;
    stepSize = 0.01;
    explore = true;
    upperConfidenceFactor = 2;
}

Tensor concat(const Tensor &a, const Tensor &b){
    return xt::concatenate(xt::xtuple(a, b));
}

double QAgent::getQ(const Tensor &state, int action) {
    return values.get(concat(state, {(double)action}))(0);
}

void QAgent::setQ(const Tensor &state, int action, double q) {
    values.set(concat(state, {(double)action}), q);
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
    double max = -INFINITY;
    int action = 0;
    for(int i = 0; i < actionCount; i++){
        auto stateAction = xt::concatenate(xt::xtuple(xt::flatten(state), Tensor((double)i)));
        double q = getQ(state, i);
        double u = upperConfidenceFactor * std::sqrt(std::log(timeStep) / (2 * visits.get(stateAction)(0)));
        if(explore){
            q += u;
        }
        if(q > max){
            max = q;
            action = i;
        }
    }

    auto stateAction = xt::concatenate(xt::xtuple(xt::flatten(state), Tensor((double)action)));
    visits.set(stateAction, visits.get(stateAction) + 1);

    return action;
}

void QAgent::trainStep(const Tensor &state, int action, double reward, double accumulativeReward, const Tensor &state2, int action2, bool isNextTerminal) {
    double delta = reward + discountFactor * maxQ(state2) - getQ(state, action);
    setQ(state, action, getQ(state, action) + stepSize * delta);
}
