//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Model.h"
#include "Differentiator.h"

Model::Model() {
    bestLoss = INFINITY;
}

Model::Model(Node &node) {
    bestLoss = INFINITY;
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

void Model::checkBest(double lossValue){
    if(lossValue < bestLoss){
        bestLoss = lossValue;
        bestParameter.clear();
        forward.eachParameter([&](Tensor &p){
            bestParameter.push_back(p);
        });
    }
}

double Model::samples(const Tensor &input, const Tensor &target, int samples) {
    Tensor output = forward.run(input, true);
    double lossValue = loss->value(output, target);

    checkBest(lossValue);

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

const Tensor &Model::predict(const Tensor &input) {
    return forward.run(input);
}

int Model::totalParameterCount() {
    int count = 0;
    forward.eachParameter([&](Tensor &p){
       count += p.size();
    });
    return count;
}

void Model::resetToBest() {
    int i = 0;
    forward.eachParameter([&](Tensor &p){
        p = bestParameter[i++];
    });
}
