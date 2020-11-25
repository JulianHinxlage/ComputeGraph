//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "LookupTable.h"

LookupTable::LookupTable() {
    defaultValue = 0;
}

const Tensor &LookupTable::get(const Tensor &input) {
    auto x = map.find(input);
    if(x != map.end()){
        return x->second;
    }else{
        return defaultValue;
    }
}

void LookupTable::set(const Tensor &input, const Tensor &output) {
    map[input] = output;
}
