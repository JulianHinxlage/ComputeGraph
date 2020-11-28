//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "ActorCriticAgent.h"
#include "model/ModelBuilder.h"

ActorCriticAgent::ActorCriticAgent() {}

void reluLayers(ModelBuilder &net, const std::vector<int> &layers){
    int in = net.outputSize;
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
}

void ActorCriticAgent::init(int inputs, int actions, const std::vector<int> &layers, const std::vector<int> &actorLayers, const std::vector<int> &criticLayers, double entropyFactor) {
    ModelBuilder net;
    net.input({inputs});

    reluLayers(net, layers);

    ModelBuilder actorNet = net;
    reluLayers(actorNet, actorLayers);
    actorNet.dense(actions);
    actorNet.softmax();

    ModelBuilder criticNet = net;
    reluLayers(criticNet, criticLayers);
    criticNet.dense(1);

    Graph graph;
    graph.add(actorNet.node);
    graph.add(criticNet.node);


    actorCritic.compile(graph);
    actorCritic.optimizer = std::make_shared<Adam>(Adam(0.001, 0.9, 0.999, 100, 0.0, 1e-8));


    Node actorTarget = Node::input({actions});
    Node criticTarget = Node::input({1});
    Node actor = Node::input({actions});
    Node critic = Node::input({1});

    Node actorLoss = -sum(actorTarget * log(actor + 0.01));
    Node criticLoss = (criticTarget - critic) * (criticTarget - critic);
    Node entropyLoss = sum(actor * log2(actor + 0.01));
    loss.compile(actorLoss + criticLoss + entropyLoss * entropyFactor);
}

int ActorCriticAgent::sampleDistribution(const Tensor &probabilities) {
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

int ActorCriticAgent::policyStep(const Tensor &state) {
    const Tensor &actor = actorCritic.forward.runMultiple({state})[0];
    int action = sampleDistribution(actor);
    return action;
}

void ActorCriticAgent::trainStep(const Tensor &state, int action, double reward, double accumulativeReward, const Tensor &state2, int action2, bool isNextTerminal) {
    auto output2 = actorCritic.forward.runMultiple({state2});
    auto &critic2 = output2[1];

    auto output = actorCritic.forward.runMultiple({state});
    auto &actor = output[0];
    auto &critic = output[1];


    Tensor criticTarget = reward + discountFactor * critic2;
    if(isNextTerminal){
        criticTarget = reward;
    }

    Tensor actorTarget = xt::zeros_like(actor);
    actorTarget(action) = criticTarget(0) - critic(0);

    loss.forward.runMultiple({actorTarget, actor, criticTarget, critic})[0](0);
    auto grads = loss.backward.runMultiple({1.0});
    actorCritic.backward.runMultiple({grads[1], grads[3](0)});
    actorCritic.updateOptimizer(1);
}
