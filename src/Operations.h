//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_OPERATIONS_H
#define COMPUTEGRAPH_OPERATIONS_H

#include "Tensor.h"

class Operations {
public:
    typedef std::function<void(Tensor &result, Tensor &lhs, Tensor &rhs)> Operation;
    static void init();
    static void add(const std::string &name, const Operation &operation);
    static const Operation &get(const std::string &name);
};


#endif //COMPUTEGRAPH_OPERATIONS_H
