//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_MODELBUILDER_H
#define COMPUTEGRAPH_MODELBUILDER_H

#include "Node.h"
#include <unordered_map>

class ModelBuilder {
public:
    Node node;
    int outputSize;
    std::unordered_map<int, Node> residualLinks;
    ModelBuilder();

    void input(const Shape &shape = {1});
    void relu();
    void sigmoid();
    void tanh();
    void softmax();
    void dense(int size);
    void dropout(double rate);
    void residual(int link = 0);
    void recurrent(int size, const std::function<void()> &activation = nullptr);
    void recurrent(int size, void (ModelBuilder::*activation)());
    void lstm(int size);
};


#endif //COMPUTEGRAPH_MODELBUILDER_H
