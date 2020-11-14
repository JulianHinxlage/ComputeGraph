//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Derivatives.h"

static std::unordered_map<std::string, Derivatives::Operation> operations;
static Derivatives::Operation identityOperation;

void Derivatives::add(const std::string &name, const Derivatives::Operation &operation) {
    operations[name] = operation;
}

const Derivatives::Operation &Derivatives::get(const std::string &name) {
    auto x = operations.find(name);
    if(x == operations.end()){
        return identityOperation;
    }else{
        return x->second;
    }
}

void Derivatives::init() {
    identityOperation = [](Node &lhsResult, Node &rhsResult, Node &lhs, Node &rhs, Node &gradient){
        lhsResult = gradient;
    };

    #define OP(name) operations[name] = [](Node &lhsResult, Node &rhsResult, Node &lhs, Node &rhs, Node &gradient)

    OP("="){
        lhsResult = gradient;
    };

    OP("s+"){
        lhsResult = gradient;
        rhsResult = gradient;
    };
    OP("s-"){
        lhsResult = gradient;
        rhsResult = gradient * -1;
    };
    OP("s*"){
        lhsResult = gradient("s*", rhs);
        rhsResult = lhs("s*", gradient);
    };
    OP("s/"){
        lhsResult = gradient / rhs;
        rhsResult = lhs / gradient;
    };

    OP("+"){
        lhsResult = gradient;
        rhsResult = gradient;
    };
    OP("-"){
        lhsResult = gradient;
        rhsResult = gradient * -1;
    };
    OP("*"){
        lhsResult = gradient * rhs.t();
        rhsResult = lhs.t() * gradient;
    };
    OP("/"){
        lhsResult = gradient / rhs;
        rhsResult = lhs / gradient;
    };
    OP("%"){
        lhsResult = gradient % rhs;
        rhsResult = lhs % gradient;
    };

    OP("t"){
        lhsResult = gradient.t();
    };
    OP(">"){
        lhsResult = gradient % (lhs > rhs);
        rhsResult = gradient % (lhs <= rhs);
    };
    OP("<"){
        lhsResult = gradient % (lhs < rhs);
        rhsResult = gradient % (lhs >= rhs);
    };
    OP(">="){
        lhsResult = gradient % (lhs >= rhs);
        rhsResult = gradient % (lhs < rhs);
    };
    OP("<="){
        lhsResult = gradient % (lhs <= rhs);
        rhsResult = gradient % (lhs > rhs);
    };

    OP("max"){
        lhsResult = gradient % (lhs > rhs);
        rhsResult = gradient % (lhs <= rhs);
    };
    OP("min"){
        lhsResult = gradient % (lhs < rhs);
        rhsResult = gradient % (lhs >= rhs);
    };

}
