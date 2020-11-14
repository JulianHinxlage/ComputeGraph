//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Sequence.h"
#include <sstream>

Sequence::Sequence() {

}

Sequence::Sequence(const Node &node) {
    generate(node);
}

void Sequence::generate(const Node &node) {
    steps.clear();
    generateStep(node);
}

const Matrix &Sequence::run(const Matrix &input) {
    std::shared_ptr<Step> output;
    for(auto &s : steps){
        output = s;
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
            case Node::OPERATION:{
                if(s->operands.size() == 2){
                    s->callback(s->value, s->operands[0]->value, s->operands[1]->value);
                }else if(s->operands.size() == 1){
                    s->callback(s->value, s->operands[0]->value, s->operands[0]->value);
                }
                break;
            }
            case Node::GRADIENT:
                break;
        }
    }
    return output->value;
}

std::shared_ptr<Sequence::Step> Sequence::generateStep(const Node &node) {
    for(auto &s : steps){
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
        case Node::OUTPUT:
        case Node::OPERATION:{
            step->operation = node.impl->operation;
            step->callback = Operations::get(step->operation);
            for(auto &n : node.impl->operands){
                step->operands.push_back(generateStep(n));
            }
            break;
        }
        case Node::GRADIENT:
            break;
    }
    steps.push_back(step);
    return step;
}

void Sequence::eachParameter(const std::function<void(Matrix &)> &callback) {
    for(auto &s : steps){
        if(s->node.impl->type == Node::PARAMETER){
            callback(s->value);
        }
    }
}

int Sequence::nodeIndex(const std::shared_ptr<Step> &step) {
    for(int i = 0; i < steps.size(); i++){
        if(steps[i] == step){
            return i;
        }
    }
    return -1;
}

template<typename T>
void join(std::stringstream &ss, const std::vector<T> &list, const std::string &delimiter){
    for(int i = 0; i < list.size(); i++){
        if(i != 0){
            ss << delimiter;
        }
        ss << list[i];
    }
}

std::string Sequence::toString() {
    std::stringstream ss;

    int index = 0;

    for(auto &s : steps){
        ss << index++ << ": ";

        switch (s->node.impl->type) {
            case Node::INPUT:
                ss << "input {";
                join(ss, s->node.impl->shape.dimensions, ", ");
                ss <<  "}" << std::endl;
                break;
            case Node::CONSTANT:
                ss << s->node.impl->values[0] << " {";
                join(ss, s->node.impl->shape.dimensions, ", ");
                ss <<  "}" << std::endl;
                break;
            case Node::PARAMETER:
                ss << "parameter {";
                join(ss, s->node.impl->shape.dimensions, ", ");
                ss <<  "}" << std::endl;
                break;
            case Node::BUFFER:
                ss << "buffer {";
                join(ss, s->node.impl->shape.dimensions, ", ");
                ss <<  "} ";
            case Node::OUTPUT:
            case Node::OPERATION:{
                if(s->operands.size() == 2){
                    ss << nodeIndex(s->operands[0]) << " " << s->operation << " " << nodeIndex(s->operands[1]) << std::endl;
                }else if(s->operands.size() == 1){
                    ss << s->operation << " " << nodeIndex(s->operands[0]) << std::endl;
                }else{
                    ss << s->operation << " [";
                    for(int i = 0; i < s->operands.size(); i++){
                        auto &o = s->operands[i];
                        if(i != 0){
                            ss << ", ";
                        }
                        ss << nodeIndex(o);
                    }
                    ss << std::endl;
                }
                break;
            }
            case Node::GRADIENT:
                break;
        }
    }

    return ss.str();
}
