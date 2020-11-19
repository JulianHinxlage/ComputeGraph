//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Operations.h"
#include <unordered_map>

static std::unordered_map<std::string, Operations::Operation> operations;
static Operations::Operation identityOperation;

void Operations::add(const std::string &name, const Operations::Operation &operation) {
    operations[name] = operation;
}

const Operations::Operation &Operations::get(const std::string &name) {
    auto x = operations.find(name);
    if(x == operations.end()){
        return identityOperation;
    }else{
        return x->second;
    }
}

void Operations::init() {
    identityOperation = [](Tensor &result, Tensor &lhs, Tensor &rhs){
        result = lhs;
    };

    #define OP(name) operations[name] = [](Tensor &result, Tensor &lhs, Tensor &rhs)

    OP("="){
        result = lhs;
    };
    OP("+="){
        result += lhs;
    };
    OP("-="){
        result -= lhs;
    };

    OP("+"){
        result = lhs + rhs;
    };
    OP("-"){
        result = lhs - rhs;
    };
    OP("/"){
        result = lhs / rhs;
    };
    OP("*"){
        result = lhs * rhs;
    };

    OP("dot"){
        result = xt::linalg::dot(lhs, rhs);
    };
    OP("t"){
        result = xt::transpose(lhs);
    };

    OP(">"){
        result = lhs > rhs;
    };
    OP("<"){
        result = lhs < rhs;
    };
    OP(">="){
        result = lhs >= rhs;
    };
    OP("<="){
        result = lhs <= rhs;
    };
    OP("max"){
        result = xt::maximum(lhs, rhs);
    };
    OP("min"){
        result = xt::minimum(lhs, rhs);
    };

    OP("exp"){
        result = xt::exp(lhs);
    };
    OP("inv"){
        result = 1.0 / lhs;
    };

}
