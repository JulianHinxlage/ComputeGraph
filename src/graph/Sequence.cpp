//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Sequence.h"

Sequence::Sequence() {}

void Sequence::setParent(const Sequence &parent) {
    parentSteps = parent.steps;
}

void Sequence::generate(const Graph &graph) {
    Operations::init();
    for(auto &node : graph.nodes){
        generateStep(node);
    }
}

std::shared_ptr<Sequence::Step> Sequence::generateStep(const Node &node) {
    for(auto &s : steps){
        if(s->node.equal(node)){
            return s;
        }
    }
    for(auto &s : parentSteps){
        if(s->node.equal(node)){
            return s;
        }
    }

    std::shared_ptr<Step> step = std::make_shared<Step>();
    step->node = node;
    step->type = node.impl->type;

    switch (node.impl->type) {
        case Node::INPUT:
            break;
        case Node::CONSTANT:
            step->value = node.impl->shape.zeros();
            if(node.impl->values.size() == 1){
                step->value = node.impl->values[0];
            }else{
                for(int i = 0; i < node.impl->values.size(); i++){
                    step->value(i) = node.impl->values[i];
                }
            }
            break;
        case Node::PARAMETER:
            step->value = node.impl->shape.zeros();
            break;
        case Node::BUFFER: {
            step->value = node.impl->shape.zeros();
            steps.push_back(step);

            step->operation = node.impl->operation;
            step->callback = Operations::get(step->operation);
            step->trainCallback = step->callback;
            if(Operations::exists(step->operation + "-train")){
                step->trainCallback = Operations::get(step->operation + "-train");
            }
            for (auto &n : node.impl->operands) {
                step->operands.push_back(generateStep(n));
            }

            for(int i = 0; i < steps.size(); i++){
                if(steps[i] == step){
                    steps.erase(steps.begin() + i);
                    break;
                }
            }
            break;
        }
        case Node::GRADIENT:
            step->value = node.impl->shape.zeros();
        case Node::OUTPUT:
        case Node::OPERATION:{
            step->operation = node.impl->operation;
            step->callback = Operations::get(step->operation);
            step->trainCallback = step->callback;
            if(Operations::exists(step->operation + "-train")){
                step->trainCallback = Operations::get(step->operation + "-train");
            }
            for(auto &n : node.impl->operands){
                step->operands.push_back(generateStep(n));
            }
            break;
        }
    }

    for(auto &s : steps){
        if(s->node.equal(node)){
            return s;
        }
    }

    steps.push_back(step);
    return step;
}

const std::vector<Tensor> &Sequence::runMultiple(const std::vector<Tensor> &inputs, bool trainMode){
    returnBuffer.clear();
    int inputIndex = 0;
    for(auto &s : steps){
        switch (s->type) {
            case Node::INPUT: {
                auto &input = inputs[inputIndex++];
                if (input.shape().size() == 1) {
                    s->value = xt::view(input, xt::all(), xt::newaxis());
                } else {
                    s->value = input;
                }
                break;
            }
            case Node::CONSTANT:
                break;
            case Node::PARAMETER:
                break;
            case Node::OUTPUT:
            case Node::BUFFER:
            case Node::GRADIENT:
            case Node::OPERATION:{
                Tensor *rhs;
                if(s->operands.size() >= 2) {
                    rhs = &s->operands[1]->value;
                }else if(s->operands.size() == 1){
                    rhs = &s->operands[0]->value;
                }
                if(s->operands.size() >= 1) {
                    if (trainMode) {
                        s->trainCallback(s->value, s->operands[0]->value, *rhs);
                    } else {
                        s->callback(s->value, s->operands[0]->value, *rhs);
                    }
                }

                if(s->type == Node::OUTPUT){
                    if(inputs[0].shape().size() == 1){
                        s->value = xt::flatten(s->value);
                    }
                    returnBuffer.push_back(s->value);
                }
                break;
            }
        }
    }
    return returnBuffer;
}

const Tensor &Sequence::run(const Tensor &input, bool trainMode) {
    return runMultiple({input}, trainMode)[0];
}

void Sequence::eachParameter(const std::function<void(Tensor &)> &callback) {
    for(auto &s : steps){
        if(s->type == Node::PARAMETER){
            callback(s->value);
        }
    }
}

void Sequence::eachGradient(const std::function<void(Tensor &, Tensor &)> &callback) {
    for(auto &s : steps){
        if(s->type == Node::GRADIENT){
            callback(s->operands[1]->value, s->value);
        }
    }
}

void Sequence::eachBuffer(const std::function<void(Tensor &buffer)> &callback){
    for(auto &s : steps){
        if(s->type == Node::BUFFER){
            callback(s->value);
        }
    }
}
