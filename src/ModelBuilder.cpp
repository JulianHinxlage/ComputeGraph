//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "ModelBuilder.h"

ModelBuilder::ModelBuilder() {
    outputSize = 0;
}

void ModelBuilder::input(const Shape &shape) {
    node = Node::input(shape);
    outputSize = shape.get(0);
}

Node relu(Node node) {
    return max(node, node * 0.01);
}

Node sigmoid(Node node) {
    return inv(exp(-node) + 1.0);
}

Node tanh(Node node) {
    return -(inv(exp(node * 2.0) + 1.0) * 2.0) + 1.0;
}

Node softmax(Node node) {
    return exp(node) / sum(exp(node));
}

void ModelBuilder::relu() {
    node = max(node, node * 0.01);
}

void ModelBuilder::sigmoid() {
    node = inv(exp(-node) + 1.0);
}

void ModelBuilder::tanh() {
    node = -(inv(exp(node * 2.0) + 1.0) * 2.0) + 1.0;
}

void ModelBuilder::softmax() {
    node = exp(node) / sum(exp(node));
}

void ModelBuilder::dense(int size) {
    node = Node::parameter({size, outputSize}).dot(node) + Node::parameter({size,1});
    outputSize = size;
}

void ModelBuilder::dropout(double rate) {
    Node mask = node("dropout", rate);
    node = node * mask;
}

void ModelBuilder::residual(int link) {
    if(residualLinks.find(link) != residualLinks.end()){
        node = node + residualLinks[link];
        residualLinks.erase(link);
    }else{
        residualLinks[link] = node;
    }
}

void ModelBuilder::recurrent(int size, const std::function<void()> &activation) {
    dense(size);
    Node buffer = Node::buffer({outputSize, 1});
    node = node + Node::parameter({outputSize, outputSize}).dot(buffer);
    if(activation){
        activation();
    }
    buffer = node;
}

void ModelBuilder::recurrent(int size, void (ModelBuilder::*activation)()) {
    recurrent(size, [&](){
        (this->*activation)();
    });
}

void ModelBuilder::lstm(int size) {
    Node cell = Node::buffer({size, 1});
    Node output = Node::buffer({size, 1});

    Node f = Node::parameter({size, outputSize}).dot(node) + Node::parameter({size,1}) + Node::parameter({size, size}).dot(output);
    Node i = Node::parameter({size, outputSize}).dot(node) + Node::parameter({size,1}) + Node::parameter({size, size}).dot(output);
    Node o = Node::parameter({size, outputSize}).dot(node) + Node::parameter({size,1}) + Node::parameter({size, size}).dot(output);
    Node c = Node::parameter({size, outputSize}).dot(node) + Node::parameter({size,1}) + Node::parameter({size, size}).dot(output);

    f = ::sigmoid(f);
    i = ::sigmoid(i);
    o = ::sigmoid(o);
    c = ::tanh(c);

    Node cell2 = f * cell + i * c;
    cell = cell2;

    node = o * cell2;
    output = node;
    outputSize = size;
}
