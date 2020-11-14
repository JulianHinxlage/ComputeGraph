//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_DIFFERENTIATOR_H
#define COMPUTEGRAPH_DIFFERENTIATOR_H

#include "Node.h"

class Differentiator {
public:
    Node differentiate(Node &node, std::vector<Node> &gradients);
private:
    Node differentiateStep(Node &node);
    void each(Node &node, const std::function<void(Node &node)> &callback);
    void each(Node &node, const std::function<void(Node &node)> &callback, std::vector<Node> &visited);

    std::vector<Node> &getGradientList(const Node &node);

    std::vector<std::pair<Node, std::vector<Node>>> gradientLists;
    std::vector<std::pair<Node, Node>> gradientResults;
};


#endif //COMPUTEGRAPH_DIFFERENTIATOR_H
