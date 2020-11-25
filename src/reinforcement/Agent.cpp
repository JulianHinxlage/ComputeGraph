//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Agent.h"

Agent::Agent() {
    discountFactor = 0.9;
    maxReplayBufferSize = 10000;
    updateAccumulativeRewards = true;
}

int Agent::step(const Tensor &state, double reward, bool terminal){
    int action = policyStep(state);

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
    return action;
}

void Agent::train(int steps){}
