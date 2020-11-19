//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Model.h"
#include "Differentiator.h"

Model::Model() {}

Model::Model(Node &node) {
    compile(node);
}

void Model::compile(Node &node) {
    forward.generate(node);
    Differentiator differentiator;
    std::vector<Node> gradients;
    Node inputGradient = differentiator.differentiate(node, gradients);
    backward.setParent(forward);
    backward.generate(inputGradient);
    for(Node &gradient : gradients){
        backward.generate(gradient);
    }

    forward.eachParameter([&](Tensor &parameter){
        parameter = xt::random::rand<double>(parameter.shape(), -1, 1);
    });
}

double Model::sample(const Tensor &input, const Tensor &target) {
    Tensor output = forward.run(input);
    Tensor gradient = lossGradient(output, target);
    double lossValue = loss(output, target);
    backward.run(gradient);
    optimizer.update([&](auto &c){backward.eachGradient(c);});
    return lossValue;
}

double Model::columnSamples(const Tensor &input, const Tensor &target, int epochs){
    double lossValue = 0;
    for(int epoch = 0; epoch < epochs; epoch++){
        lossValue = 0;
        for(int i = 0; i < input.shape(0); i++){
            lossValue += sample(xt::view(input, i), xt::view(target, i));
        }
    }
    return lossValue;
}

double Model::loss(const Tensor &output, const Tensor &target) {
    return xt::sum((output - target) * (output - target))() / output.size();
}

Tensor Model::lossGradient(const Tensor &output, const Tensor &target) {
    return (output - target) / output.size();
}
