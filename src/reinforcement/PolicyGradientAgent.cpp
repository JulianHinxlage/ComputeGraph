//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "PolicyGradientAgent.h"
#include "model/ModelBuilder.h"

PolicyGradientAgent::PolicyGradientAgent() {
    baseline.maxValues = 1000;
    variance.maxValues = 1000;
}

void PolicyGradientAgent::init(int inputs, int actions, const std::vector<int> &layers) {
    ModelBuilder net;
    net.input({inputs});

    int in = inputs;
    for(int i : layers){
        if(i == in){
            net.residual();
        }
        net.dense(i);
        net.relu();
        if(i == in){
            net.residual();
        }
        in = i;
    }

    net.dense(actions);
    net.softmax();

    policy.compile(net.node);
    policy.optimizer = std::make_shared<Adam>();
    policy.optimizer->batchSize = 100;

    Node rewards = Node::input({actions});
    Node probabilities = Node::input({actions});
    Node actionLoss = sum(rewards * log(probabilities + 0.01));
    Node entropyLoss = -sum(probabilities * log(probabilities + 0.01));
    loss.compile(actionLoss + entropyLoss * 0.01);
}

int PolicyGradientAgent::sampleDistribution(const Tensor &probabilities) {
    double value = xt::random::rand<double>({1}, 0, 1)(0);
    int outcome = 0;
    for(int i = 0; i < probabilities.size(); i++){
        outcome = i;
        if(value > probabilities[i]){
            value -= probabilities[i];
        }else{
            break;
        }
    }
    return outcome;
}

int PolicyGradientAgent::policyStep(const Tensor &state) {
    const Tensor &actionProbabilities = policy.forward.run(state);
    int action = sampleDistribution(actionProbabilities);
    return action;
}

void PolicyGradientAgent::trainStep(const Tensor &state, int action, double reward, double accumulativeReward, const Tensor &state2, int action2) {
    const Tensor &actionProbabilities = policy.forward.run(state);
    Tensor values = xt::zeros_like(actionProbabilities);

    baseline.add(accumulativeReward);
    variance.add(accumulativeReward * accumulativeReward);
    double b = baseline.mean();
    double v = variance.mean() - b * b;

    values(action) = (accumulativeReward - b) / std::sqrt(v + 0.0001);
    loss.forward.runMultiple({values, actionProbabilities})[0](0);
    policy.backward.run(loss.backward.runMultiple({-0.01})[1]);
    policy.updateOptimizer(1);
}
