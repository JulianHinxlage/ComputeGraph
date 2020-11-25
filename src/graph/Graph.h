//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_GRAPH_H
#define COMPUTEGRAPH_GRAPH_H

#include "Node.h"

class Graph {
public:
    std::vector<Node> nodes;
    Graph();
    Graph(const Node &node);
    void add(const Node &node);
};


#endif //COMPUTEGRAPH_GRAPH_H
