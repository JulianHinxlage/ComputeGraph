//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_DERIVATIVES_H
#define COMPUTEGRAPH_DERIVATIVES_H

#include "Operations.h"
#include "Node.h"

class Derivatives {
public:
    typedef std::function<void(Node &lhsResult, Node &rhsResult, Node &lhs, Node &rhs, Node &gradient)> Operation;
    static void init();
    static void add(const std::string &name, const Operation &operation);
    static const Operation &get(const std::string &name);
};


#endif //COMPUTEGRAPH_DERIVATIVES_H
