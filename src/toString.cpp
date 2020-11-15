//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "toString.h"
#include <sstream>

int nodeIndex(Sequence &sequence, const std::shared_ptr<Sequence::Step> &step) {
    for(int i = 0; i < sequence.steps.size(); i++){
        if(sequence.steps[i] == step){
            return i + 1;
        }
    }

    for(int i = 0; i < sequence.parentSteps.size(); i++){
        if(sequence.parentSteps[i] == step){
            return -i - 1;
        }
    }

    return 0;
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

std::string toString(Sequence &sequence) {
    std::stringstream ss;

    int index = 1;

    for(auto &s : sequence.steps){
        ss << index++ << ": ";

        switch (s->node.impl->type) {
            case Node::INPUT:
                ss << "input: {";
                join(ss, s->node.impl->shape.dimensions, ", ");
                ss <<  "}" << std::endl;
                break;
            case Node::CONSTANT:
                ss << "constant: " << s->node.impl->values[0] << " {";
                join(ss, s->node.impl->shape.dimensions, ", ");
                ss <<  "}" << std::endl;
                break;
            case Node::PARAMETER:
                ss << "parameter: {";
                join(ss, s->node.impl->shape.dimensions, ", ");
                ss <<  "}" << std::endl;
                break;
            case Node::GRADIENT:
                if(s->operands.size() == 2){
                    ss << "gradient(" << nodeIndex(sequence, s->operands[1]) << "): " << nodeIndex(sequence, s->operands[0]) << std::endl;
                }
                break;
            case Node::BUFFER:
                ss << "buffer: {";
                join(ss, s->node.impl->shape.dimensions, ", ");
                ss <<  "} ";
            case Node::OUTPUT:
                if(s->node.impl->type == Node::OUTPUT){
                    ss << "output: ";
                }
            case Node::OPERATION:{
                if(s->operands.size() == 2){
                    ss << nodeIndex(sequence, s->operands[0]) << " " << s->operation << " " << nodeIndex(sequence, s->operands[1]) << std::endl;
                }else if(s->operands.size() == 1){
                    ss << s->operation << " " << nodeIndex(sequence, s->operands[0]) << std::endl;
                }else{
                    ss << s->operation << " [";
                    for(int i = 0; i < s->operands.size(); i++){
                        auto &o = s->operands[i];
                        if(i != 0){
                            ss << ", ";
                        }
                        ss << nodeIndex(sequence, o);
                    }
                    ss << std::endl;
                }
                break;
            }
        }
    }

    return ss.str();
}
