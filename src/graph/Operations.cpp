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

bool Operations::exists(const std::string &name) {
    return operations.find(name) != operations.end();
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
        //todo: tensor rank agnostic
        for(int r = 0; r < lhs.shape(0); r++){
            for(int c = 0; c < lhs.shape(0); c++){
                result(r % result.shape(0), c % result.shape(1)) += lhs(r % lhs.shape(0), c % lhs.shape(1));
            }
        }
    };
    OP("-="){
        //todo: tensor rank agnostic
        for(int r = 0; r < lhs.shape(0); r++){
            for(int c = 0; c < lhs.shape(0); c++){
                result(r % result.shape(0), c % result.shape(1)) -= lhs(r % lhs.shape(0), c % lhs.shape(1));
            }
        }
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

    OP("concat"){
        result = xt::concatenate(xt::xtuple(lhs, rhs));
    };
    OP("split-left"){
        auto shape = rhs.shape();
        if(shape.size() == 1){
            result = xt::view(lhs, xt::range(0, shape[0]));
        }else if(shape.size() == 2){
            result = xt::view(lhs, xt::range(0, shape[0]), xt::all());
        }
    };
    OP("split-right"){
        auto shape = rhs.shape();
        if(shape.size() == 1){
            result = xt::view(lhs, xt::range(shape[0],lhs.shape(0)));
        }else if(shape.size() == 2){
            result = xt::view(lhs, xt::range(shape[0],lhs.shape(0)), xt::all());
        }
    };

    OP("exp"){
        result = xt::exp(lhs);
    };
    OP("inv"){
        result = 1.0 / lhs;
    };
    OP("sum"){
        result = xt::sum(lhs);
    };
    OP("random"){
        result = xt::random::rand<double>(lhs.shape());
    };
    OP("dropout"){
        result = xt::ones<double>(lhs.shape()) * (1.0 - rhs);
    };
    OP("dropout-train"){
        result = xt::random::rand<double>(lhs.shape()) > rhs;
    };

}
