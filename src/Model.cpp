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

    Graph gradientGraph = differentiator.differentiate(node);
    backward.setParent(forward);
    backward.generate(gradientGraph);

    forward.eachParameter([&](Tensor &parameter){
        parameter = xt::random::rand<double>(parameter.shape(), -1, 1);
    });
}

const Tensor &Model::predict(const Tensor &input) {
    return forward.run(input);
}

const Tensor &Model::gradient(const Tensor &gradient) {
    auto &output = backward.run(gradient);
    optimizer->update([&](auto &c){backward.eachGradient(c);}, 1);
    return output;
}

void Model::updateOptimizer(int samples){
    optimizer->update([&](auto &c){backward.eachGradient(c);}, samples);
}



double Model::fit(const Tensor &input, const Tensor &target, int samples, int epochs) {
    double loss = 0;
    for(int epoch = 0; epoch < epochs; epoch++) {
        Tensor output = forward.run(input, true);
        loss = lossFunction->value(output, target);
        Tensor gradient = lossFunction->gradient(output, target);
        backward.run(gradient);
        optimizer->update([&](auto &c) { backward.eachGradient(c); }, samples);
    }
    return loss;
}

double Model::fitColumns(const Tensor &input, const Tensor &target, int epochs) {
    double loss = 0;
    for(int epoch = 0; epoch < epochs; epoch++) {
        loss = 0;
        for(int sample = 0; sample < input.shape(1); sample++) {
            auto sampleInput = xt::view(input, xt::all(), sample, xt::newaxis());
            auto sampleTarget = xt::view(target, xt::all(), sample, xt::newaxis());

            Tensor output = forward.run(sampleInput, true);
            loss += lossFunction->value(output, sampleTarget);
            Tensor gradient = lossFunction->gradient(output, sampleTarget);
            backward.run(gradient);
            optimizer->update([&](auto &c) { backward.eachGradient(c); }, 1);
        }
    }
    return loss;
}

int Model::totalParameterCount() {
    int count = 0;
    forward.eachParameter([&](Tensor &p){
        count += p.size();
    });
    return count;
}
