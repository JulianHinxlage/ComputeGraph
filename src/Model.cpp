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

double Model::samples(const Tensor &input, const Tensor &target, int samples) {
    Tensor output = forward.run(input);
    double lossValue = loss->value(output, target);
    Tensor gradient = loss->gradient(output, target);
    backward.run(gradient);
    optimizer->update([&](auto &c){backward.eachGradient(c);}, samples);
    return lossValue;
}

double Model::columnSamples(const Tensor &input, const Tensor &target, int epochs){
    double lossValue = 0;
    for(int epoch = 0; epoch < epochs; epoch++){
        lossValue = 0;
        for(int i = 0; i < input.shape(1); i++){
            lossValue += samples(xt::view(input, xt::all(), i, xt::newaxis()), xt::view(target, xt::all(), i, xt::newaxis()));
        }
    }
    return lossValue;
}

int Model::totalParameterCount() {
    int count = 0;
    forward.eachParameter([&](Tensor &p){
       count += p.size();
    });
    return count;
}
