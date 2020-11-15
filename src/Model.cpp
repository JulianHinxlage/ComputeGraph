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

    forward.eachParameter([&](Matrix &parameter){
       parameter.setRandom();
    });
}

double Model::sample(const Matrix &input, const Matrix &target) {
    Matrix output = forward.run(input);
    Matrix gradient = lossGradient(output, target);
    double lossValue = loss(output, target);
    backward.run(gradient);
    optimizer.update([&](auto &c){backward.eachGradient(c);});
    return lossValue;
}

double Model::columnSamples(const Matrix &input, const Matrix &target, int epochs){
    double lossValue = 0;
    for(int epoch = 0; epoch < epochs; epoch++){
        lossValue = 0;
        for(int i = 0; i < input.cols(); i++){
            lossValue += sample(input.col(i), target.col(i));
        }
    }
    return lossValue;
}

double Model::loss(const Matrix &output, const Matrix &target) {
    return (output - target).cwiseProduct(output - target).sum() / output.cols() / output.rows();
}

Matrix Model::lossGradient(const Matrix &output, const Matrix &target) {
    return (output - target) / output.cols() / output.rows();
}
