//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Node.h"

Node::Impl::Impl() {
    type = NONE;
    shape = {};
    values = {};
    isScalar = false;
}

Node::Node() {
    impl = std::make_shared<Impl>();
}

Node Node::input(const Shape &shape) {
    Node node;
    node.impl->type = INPUT;
    node.impl->shape = shape;
    return node;
}

Node Node::parameter(const Shape &shape) {
    Node node;
    node.impl->type = PARAMETER;
    node.impl->shape = shape;
    return node;
}

Node Node::buffer(const Shape &shape) {
    Node node;
    node.impl->type = BUFFER;
    node.impl->shape = shape;
    return node;
}

Node Node::constant(const Shape &shape, const std::vector<double> &values) {
    Node node;
    node.impl->type = CONSTANT;
    node.impl->shape = shape;
    node.impl->values = values;
    node.impl->isScalar = (shape == Shape(1));
    return node;
}

Node Node::constant(const Shape &shape, double value) {
    return constant(shape, (std::vector<double>){value});
}

Node Node::constant(double value) {
    return constant({1}, value);
}

Node Node::operator()(const std::string &operation) const {
    Node node;
    node.impl->type = OPERATION;
    node.impl->operation = operation;
    node.impl->operands.push_back(*this);
    return node;
}

Node Node::operator()(const std::string &operation, const Node &rhs) const {
    Node node;
    node.impl->type = OPERATION;
    node.impl->operation = operation;
    node.impl->operands.push_back(*this);
    node.impl->operands.push_back(rhs);
    return node;
}

Node Node::operator()(const std::string &operation, double rhs) const {
    return operator()(operation, Node::constant(rhs));
}

Node Node::operator+(const Node &rhs) const {
    return operator()("+", rhs);
}

Node Node::operator-(const Node &rhs) const {
    return operator()("-", rhs);
}

Node Node::operator*(const Node &rhs) const {
    return operator()("*", rhs);
}

Node Node::operator/(const Node &rhs) const {
    return operator()("/", rhs);
}

Node Node::dot(const Node &rhs) const {
    return operator()("dot", rhs);
}

Node Node::operator+(double rhs) const {
    return operator()("+", rhs);
}

Node Node::operator-(double rhs) const {
    return operator()("-", rhs);
}

Node Node::operator*(double rhs) const {
    return operator()("*", rhs);
}

Node Node::operator/(double rhs) const {
    return operator()("/", rhs);
}

Node Node::operator>(const Node &rhs) const {
    return operator()(">", rhs);
}

Node Node::operator<(const Node &rhs) const {
    return operator()("<", rhs);
}

Node Node::operator>=(const Node &rhs) const {
    return operator()(">=", rhs);
}

Node Node::operator<=(const Node &rhs) const {
    return operator()("<=", rhs);
}

Node Node::operator-() const {
    return *this * -1.0;
}

Node Node::operator+() const {
    return *this;
}

Node Node::t() {
    return operator()("t");
}

Node &Node::operator=(const Node &rhs) {
    if(impl->type == BUFFER){
        impl->operation = "=";
        impl->operands.push_back(rhs);
    }else{
        impl = rhs.impl;
    }
    return *this;
}

bool Node::operator==(const Node &rhs) const {
    return impl == rhs.impl;
}

bool Node::equal(const Node &rhs) {
    if(impl == rhs.impl){
        return true;
    }
    if(impl->type != rhs.impl->type){
        return false;
    }
    switch (impl->type) {
        case NONE:
            return true;
        case INPUT:
        case PARAMETER:
            return false;
        case CONSTANT:
            return impl->shape == rhs.impl->shape && impl->values == rhs.impl->values;
        case OUTPUT:
        case BUFFER:
        case OPERATION:{
            if(impl->operation != rhs.impl->operation){
                return false;
            }
            if(impl->operands.size() != rhs.impl->operands.size()){
                return false;
            }
            for(int i = 0; i < impl->operands.size(); i++){
                if(!impl->operands[i].equal(rhs.impl->operands[i])){
                    return false;
                }
            }
            return true;
        }
    }
    return false;
}

Node Node::copy() {
    Node node;
    node.impl->type = impl->type;
    node.impl->shape = impl->shape;
    node.impl->values = impl->values;
    node.impl->operation = impl->operation;
    for(int i = 0; i < impl->operands.size(); i++){
        node.impl->operands.push_back(impl->operands[i].copy());
    }
    return node;
}

Node dot(const Node &lhs, const Node &rhs){
    return lhs.dot(rhs);
}

Node max(const Node &lhs, const Node &rhs){
    return lhs("max", rhs);
}

Node min(const Node &lhs, const Node &rhs){
    return lhs("min", rhs);
}

Node ternary(const Node &condition, const Node &lhs, const Node &rhs) {
    return (condition * lhs) + ((condition * -1.0f + 1.0f) * rhs);
}

Node exp(const Node &lhs){
    return lhs("exp");
}

Node inv(const Node &lhs){
    return lhs("inv");
}

Node sum(const Node &lhs) {
    return lhs("sum");
}

Node log(const Node &lhs){
    return lhs("log");
}
