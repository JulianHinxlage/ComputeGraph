//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Sequence.h"

Sequence::Sequence() {}

void Sequence::setParent(const Sequence &parent) {
    parentSteps = parent.steps;
}

void Sequence::generate(const Node &node) {
    if(node.impl->type == Node::OPERATION){
        node.impl->type = Node::OUTPUT;
    }
    generateStep(node);
}

std::shared_ptr<Sequence::Step> Sequence::generateStep(const Node &node) {
    for(auto &s : steps){
        if(s->node == node){
            return s;
        }
    }
    for(auto &s : parentSteps){
        if(s->node == node){
            return s;
        }
    }

    std::shared_ptr<Step> step = std::make_shared<Step>();
    step->node = node;

    switch (node.impl->type) {
        case Node::INPUT:
            break;
        case Node::CONSTANT:
            step->value = node.impl->shape.zeros();
            if(node.impl->values.size() == 1){
                step->value.setConstant(node.impl->values[0]);
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
            int bufferStepIndex = steps.size();
            steps.push_back(step);

            step->operation = node.impl->operation;
            step->callback = Operations::get(step->operation);
            for (auto &n : node.impl->operands) {
                step->operands.push_back(generateStep(n));
            }

            steps.erase(steps.begin() + bufferStepIndex);
            break;
        }
        case Node::GRADIENT:
            step->value = node.impl->shape.zeros();
        case Node::OUTPUT:
        case Node::OPERATION:{
            step->operation = node.impl->operation;
            step->callback = Operations::get(step->operation);
            for(auto &n : node.impl->operands){
                step->operands.push_back(generateStep(n));
            }
            break;
        }
    }
    steps.push_back(step);
    return step;
}

const Matrix &Sequence::run(const Matrix &input) {
    std::shared_ptr<Step> output;
    for(auto &s : steps){
        switch (s->node.impl->type) {
            case Node::INPUT:
                s->value = input;
                break;
            case Node::CONSTANT:
                break;
            case Node::PARAMETER:
                break;
            case Node::OUTPUT:
                output = s;
            case Node::BUFFER:
            case Node::GRADIENT:
            case Node::OPERATION:{
                if(s->operands.size() == 2){
                    s->callback(s->value, s->operands[0]->value, s->operands[1]->value);
                }else if(s->operands.size() == 1){
                    s->callback(s->value, s->operands[0]->value, s->operands[0]->value);
                }
                break;
            }
        }
    }
    if(output){
        return output->value;
    }else{
        return steps.back()->value;
    }
}

void Sequence::eachParameter(const std::function<void(Matrix &)> &callback) {
    for(auto &s : steps){
        if(s->node.impl->type == Node::PARAMETER){
            callback(s->value);
        }
    }
}

void Sequence::eachGradient(const std::function<void(Matrix &, Matrix &)> &callback) {
    for(auto &s : steps){
        if(s->node.impl->type == Node::GRADIENT){
            callback(s->operands[1]->value, s->value);
        }
    }
}
