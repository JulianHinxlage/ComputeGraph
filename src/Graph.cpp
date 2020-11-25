//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Graph.h"

Graph::Graph() {}

Graph::Graph(const Node &node) {
    add(node);
}

void Graph::add(const Node &node) {
    if(node.impl->type == Node::OPERATION){
        Node n;
        n.impl->type = Node::OUTPUT;
        n.impl->operands.push_back(node);
        n.impl->operation = "=";
        //node.impl->type = Node::OUTPUT;
        nodes.push_back(n);
    }else{
        nodes.push_back(node);
    }
}
