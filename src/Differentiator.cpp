//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Differentiator.h"
#include "Derivatives.h"

Node Differentiator::differentiate(Node &node, std::vector<Node> &gradients) {
    gradientLists.clear();
    gradientResults.clear();

    //clear usage
    each(node, [](Node &node){
       node.impl->usage.clear();
    });

    //update usage
    each(node, [](Node &node){
        for(auto &n : node.impl->operands){
            n.impl->usage.push_back(node);
        }
    });

    std::vector<Node> inputGradients;
    each(node, [&](Node &node){
        if(node.impl->type == Node::INPUT){
            Node n =differentiateStep(node);
            n.impl->type = Node::OUTPUT;
            inputGradients.push_back(n);
        }else if(node.impl->type == Node::PARAMETER){
            Node n;
            n.impl->type = Node::GRADIENT;
            n.impl->operands.push_back(differentiateStep(node));
            n.impl->operands.push_back(node);
            n.impl->shape = node.impl->shape;
            n.impl->operation = "+=";
            gradients.push_back(n);
        }
    });

    return inputGradients[0];
}

Node Differentiator::differentiateStep(Node &node) {
    for(auto &g : gradientResults){
        if(node == g.first){
            return g.second;
        }
    }

    for(auto &n : node.impl->usage){
        differentiateStep(n);
    }

    Node gradient;

    auto &list = getGradientList(node);
    for(int i = 0; i < list.size(); i++){
        auto &n = list[i];
        if(i == 0){
            gradient = n;
        }else{
            gradient = gradient + n;
        }
    }


    if(node.impl->usage.size() == 0){
        gradient = Node::input();
    }

    switch (node.impl->type) {
        case Node::INPUT:
             break;
        case Node::CONSTANT:
            break;
        case Node::PARAMETER:
            break;
        case Node::OUTPUT:
        case Node::BUFFER:
        case Node::OPERATION: {
            Node lhs = node.impl->operands.size() > 0 ? node.impl->operands[0] : Node();
            Node rhs = node.impl->operands.size() > 1 ? node.impl->operands[1] : lhs;

            Node lhsResult;
            Node rhsResult;

            Derivatives::get(node.impl->operation)(lhsResult, rhsResult, lhs, rhs, gradient);

            if(node.impl->operands.size() > 0) {
                getGradientList(lhs).push_back(lhsResult);
            }
            if(node.impl->operands.size() > 1){
                getGradientList(rhs).push_back(rhsResult);
            }
            break;
        }
        case Node::GRADIENT:
            break;
    }

    gradientResults.emplace_back(node, gradient);
    return gradient;
}

void Differentiator::each(Node &node, const std::function<void(Node &)> &callback) {
    std::vector<Node> visited;
    each(node, callback, visited);
}

void Differentiator::each(Node &node, const std::function<void(Node &)> &callback, std::vector<Node> &visited) {
    for(auto &n : visited){
        if(n == node){
            return;
        }
    }

    callback(node);
    visited.push_back(node);
    for(auto &n : node.impl->operands){
        each(n, callback, visited);
    }
}

std::vector<Node> &Differentiator::getGradientList(const Node &node) {
    for(auto &l : gradientLists){
        if(l.first == node){
            return l.second;
        }
    }
    gradientLists.push_back({node, {}});
    return gradientLists.back().second;
}
