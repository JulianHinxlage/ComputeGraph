//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_SEQUENCE_H
#define COMPUTEGRAPH_SEQUENCE_H

#include "Node.h"
#include "Operations.h"

class Sequence {
public:
    Sequence();
    void setParent(const Sequence &parent);
    void generate(const Node &node);
    const Tensor &run(const Tensor &input);
    void eachParameter(const std::function<void(Tensor &parameter)> &callback);
    void eachGradient(const std::function<void(Tensor &parameter, Tensor &gradient)> &callback);
    void eachBuffer(const std::function<void(Tensor &buffer)> &callback);

    class Step{
    public:
        std::string operation;
        std::vector<std::shared_ptr<Step>> operands;
        Node node;
        Node::Type type;
        Operations::Operation callback;
        Tensor value;
    };

    std::vector<std::shared_ptr<Step>> steps;
    std::vector<std::shared_ptr<Step>> parentSteps;

private:
    std::shared_ptr<Step> generateStep(const Node &node);
};


#endif //COMPUTEGRAPH_SEQUENCE_H
