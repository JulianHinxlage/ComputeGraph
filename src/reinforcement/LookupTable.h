//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_LOOCKUPTABLE_H
#define COMPUTEGRAPH_LOOCKUPTABLE_H

#include "graph/Tensor.h"
#include <unordered_map>

struct TensorCompare{
    bool operator()(const Tensor &a, const Tensor &b) const{
        if(a.shape().size() < b.shape().size()){
            return true;
        }else if(a.shape().size() > b.shape().size()){
            return false;
        }
        for(int i = 0; i < a.shape().size(); i++){
            if(a.shape(i) < b.shape(i)){
                return true;
            }else if(a.shape(i) > b.shape(i)){
                return false;
            }
        }

        for(int i = 0; i < a.size(); i++){
            if(a(i) < b(i)){
                return true;
            }else if(a(i) > b(i)){
                return false;
            }
        }
        return false;
    }
};

class LookupTable {
public:
    std::map<Tensor, Tensor, TensorCompare> map;
    Tensor defaultValue;
    LookupTable();
    const Tensor &get(const Tensor &input);
    void set(const Tensor &input, const Tensor &output);
};


#endif //COMPUTEGRAPH_LOOCKUPTABLE_H
