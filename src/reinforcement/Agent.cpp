//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Agent.h"

Agent::Agent() {
    discountFactor = 0.9;
    maxReplayBufferSize = 10000;
    updateAccumulativeRewards = true;
    onPolicyTrain = false;
    timeStep = 0;
}

int Agent::step(const Tensor &state, double reward, bool terminal){
    int action = policyStep(state);
    timeStep++;

    replayBuffer.push_back({
        state, action, reward, 0, terminal
    });

    if(replayBuffer.size() >= maxReplayBufferSize){
        replayBuffer.erase(replayBuffer.begin());
    }

    if(updateAccumulativeRewards) {
        //update accumulative rewards of the episode
        double factor = 1;
        for (int i = replayBuffer.size() - 1; i >= 0; i--) {
            if (!replayBuffer[i].terminal || i == replayBuffer.size() - 1) {
                replayBuffer[i].accumulativeReward += reward * factor;
                factor *= discountFactor;
            } else {
                break;
            }
        }
    }

    if(onPolicyTrain && replayBuffer.size() >= 2){
        int index = replayBuffer.size() - 2;
        if(!replayBuffer[index].terminal) {
            auto &state = replayBuffer[index].state;
            auto &action = replayBuffer[index].action;
            auto &reward = replayBuffer[index + 1].reward;
            auto &accumulativeReward = replayBuffer[index+1].accumulativeReward;
            auto &state2 = replayBuffer[index + 1].state;
            auto &action2 = replayBuffer[index + 1].action;
            auto &isNextTerminal = replayBuffer[index + 1].terminal;
            trainStep(state, action, reward, accumulativeReward, state2, action2, isNextTerminal);
        }
    }

    return action;
}

void Agent::trainAll() {
    for(int i = 0; i < replayBuffer.size() - 1; i++){
        if(replayBuffer[i].terminal){
            continue;
        }
        auto &state = replayBuffer[i].state;
        auto &action = replayBuffer[i].action;
        auto &reward = replayBuffer[i+1].reward;
        auto &accumulativeReward = replayBuffer[i+1].accumulativeReward;
        auto &state2 = replayBuffer[i+1].state;
        auto &action2 = replayBuffer[i+1].action;
        auto &isNextTerminal = replayBuffer[i+1].terminal;

        trainStep(state, action, reward, accumulativeReward, state2, action2, isNextTerminal);
    }
}

void Agent::train(int steps) {
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
        auto &accumulativeReward = replayBuffer[index+1].accumulativeReward;
        auto &state2 = replayBuffer[index+1].state;
        auto &action2 = replayBuffer[index+1].action;
        auto &isNextTerminal = replayBuffer[index+1].terminal;

        trainStep(state, action, reward, accumulativeReward, state2, action2, isNextTerminal);
    }
}

void Agent::trainStep(const Tensor &state, int action, double reward, double accumulativeReward, const Tensor &state2, int action2, bool isNextTerminal) {}

