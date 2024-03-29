//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Derivatives.h"

static std::unordered_map<std::string, Derivatives::Operation> operations;
static Derivatives::Operation identityOperation;
static bool initedDerivatives = false;

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
    if(initedDerivatives){
        return;
    }
    initedDerivatives = true;

    identityOperation = [](Node &lhsResult, Node &rhsResult, Node &lhs, Node &rhs, Node &gradient){
        lhsResult = gradient;
    };

    #define OP(name) operations[name] = [](Node &lhsResult, Node &rhsResult, Node &lhs, Node &rhs, Node &gradient)

    OP("="){
        lhsResult = gradient;
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
        lhsResult = gradient * rhs;
        rhsResult = lhs * gradient;
    };
    OP("/"){
        lhsResult = gradient / rhs;
        rhsResult = -inv(rhs * rhs) * lhs * gradient;
    };
    OP("dot"){
        lhsResult = gradient.dot(rhs.t());
        rhsResult = lhs.t().dot(gradient);
    };

    OP("t"){
        lhsResult = gradient.t();
    };
    OP(">"){
        lhsResult = gradient * (lhs > rhs);
        rhsResult = gradient * (lhs <= rhs);
    };
    OP("<"){
        lhsResult = gradient * (lhs < rhs);
        rhsResult = gradient * (lhs >= rhs);
    };
    OP(">="){
        lhsResult = gradient * (lhs >= rhs);
        rhsResult = gradient * (lhs < rhs);
    };
    OP("<="){
        lhsResult = gradient * (lhs <= rhs);
        rhsResult = gradient * (lhs > rhs);
    };

    OP("concat"){
        lhsResult = gradient("split-left", lhs);
        rhsResult = gradient("split-right", rhs);
    };

    OP("max"){
        lhsResult = gradient * (lhs > rhs);
        rhsResult = gradient * (lhs <= rhs);
    };
    OP("min"){
        lhsResult = gradient * (lhs < rhs);
        rhsResult = gradient * (lhs >= rhs);
    };
    OP("exp"){
        lhsResult = exp(lhs) * gradient;
    };
    OP("inv"){
        lhsResult = -inv(lhs * lhs) * gradient;
    };
    OP("log"){
        lhsResult = gradient / lhs;
    };
    OP("log2"){
        lhsResult = gradient / (lhs * std::log(2));
    };
    OP("sum"){
        lhsResult = gradient;
    };
    OP("random"){
        lhsResult = gradient * 0;
    };
    OP("dropout"){
        lhsResult = gradient * 0;
    };

}
