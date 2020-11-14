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
    identityOperation = [](Matrix &result, Matrix &lhs, Matrix &rhs){
        result = lhs;
    };

    #define OP(name) operations[name] = [](Matrix &result, Matrix &lhs, Matrix &rhs)

    OP("="){
        result = lhs;
    };

    OP("s+"){
        result = lhs.unaryExpr([&](double a) {
            return a + rhs(0);
        });
    };
    OP("s-"){
        result = lhs.unaryExpr([&](double a) {
            return a - rhs(0);
        });
    };
    OP("s*"){
        result = lhs.unaryExpr([&](double a) {
            return a * rhs(0);
        });
    };
    OP("s/"){
        result = lhs.unaryExpr([&](double a) {
            return a * rhs(0);
        });
    };

    OP("+"){
        result = lhs + rhs;
    };
    OP("-"){
        result = lhs - rhs;
    };
    OP("*"){
        result = lhs * rhs;
    };
    OP("/"){
        result = lhs.cwiseProduct(rhs.cwiseInverse());
    };
    OP("%"){
        result = lhs.cwiseProduct(rhs);
    };
    OP("t"){
        result = lhs.transpose();
    };
    OP(">"){
        result = lhs.binaryExpr(rhs, [](double a, double b) -> double{return a > b;});
    };
    OP("<"){
        result = lhs.binaryExpr(rhs, [](double a, double b) -> double{return a < b;});
    };
    OP(">="){
        result = lhs.binaryExpr(rhs, [](double a, double b) -> double{return a >= b;});
    };
    OP("<="){
        result = lhs.binaryExpr(rhs, [](double a, double b) -> double{return a <= b;});
    };

    OP("+="){
        result += lhs;
    };
    OP("-="){
        result -= lhs;
    };
    OP("max"){
        result = lhs.cwiseMax(rhs);
    };
    OP("min"){
        result = lhs.cwiseMin(rhs);
    };

}
