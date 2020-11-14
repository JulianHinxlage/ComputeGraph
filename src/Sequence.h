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
    Sequence(const Node &node);
    void generate(const Node &node);
    const Matrix &run(const Matrix &input);
    std::string toString();
    void eachParameter(const std::function<void(Matrix &)> &callback);

private:
    class Step{
    public:
        std::string operation;
        std::vector<std::shared_ptr<Step>> operands;
        Node node;
        Operations::Operation callback;
        Matrix value;
    };

    std::vector<std::shared_ptr<Step>> steps;

    std::shared_ptr<Step> generateStep(const Node &node);
    int nodeIndex(const std::shared_ptr<Step> &step);
};


#endif //COMPUTEGRAPH_SEQUENCE_H
