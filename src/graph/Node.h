//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_NODE_H
#define COMPUTEGRAPH_NODE_H

#include "Shape.h"
#include <memory>

class Node {
public:
    Node();

    static Node input(const Shape &shape = {1});
    static Node parameter(const Shape &shape = {1});
    static Node buffer(const Shape &shape = {1});
    static Node constant(const Shape &shape, const std::vector<double> &values);
    static Node constant(const Shape &shape, double value = 1);
    static Node constant(double value);

    Node operator()(const std::string &operation) const;
    Node operator()(const std::string &operation, const Node &rhs) const;
    Node operator()(const std::string &operation, double rhs) const;

    Node operator+(const Node &rhs) const;
    Node operator-(const Node &rhs) const;
    Node operator*(const Node &rhs) const;
    Node operator/(const Node &rhs) const;
    Node dot(const Node &rhs) const;

    Node operator+(double rhs) const;
    Node operator-(double rhs) const;
    Node operator*(double rhs) const;
    Node operator/(double rhs) const;

    Node operator>(const Node &rhs) const;
    Node operator<(const Node &rhs) const;
    Node operator>=(const Node &rhs) const;
    Node operator<=(const Node &rhs) const;

    Node operator-() const;
    Node operator+() const;
    Node t();

    Node &operator=(const Node &rhs);
    bool operator==(const Node &rhs) const;
    bool equal(const Node &rhs);
    Node copy();

    enum Type{
        NONE,
        INPUT,
        PARAMETER,
        CONSTANT,
        OPERATION,
        BUFFER,
        OUTPUT,
        GRADIENT,
    };

    class Impl{
    public:
        Type type;
        Shape shape;
        std::vector<double> values;
        std::string operation;
        std::vector<Node> operands;
        std::vector<Node> usage;
        bool isScalar;
        Impl();
    };

    std::shared_ptr<Impl> impl;
};

Node dot(const Node &lhs, const Node &rhs);
Node max(const Node &lhs, const Node &rhs);
Node min(const Node &lhs, const Node &rhs);
Node ternary(const Node &condition, const Node &lhs, const Node &rhs);
Node exp(const Node &lhs);
Node inv(const Node &lhs);
Node sum(const Node &lhs);
Node log(const Node &lhs);
Node log2(const Node &lhs);

#endif //COMPUTEGRAPH_NODE_H
