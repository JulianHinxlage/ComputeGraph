//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "ModelState.h"

ModelState::ModelState() {
    value = 0;
    hasValue = false;
}

void ModelState::load(Model &model) {
    int index = 0;
    model.forward.eachParameter([&](Tensor &p){
        if(index <= parameter.size()){
            p = parameter[index++];
        }
    });
}

void ModelState::save(Model &model) {
    int index = 0;
    model.forward.eachParameter([&](Tensor &p){
        if(index <= parameter.size()){
            parameter.resize(index+1);
        }
        parameter[index++] = p;
    });
}

void ModelState::saveOnMax(Model &model, double value) {
    if(value > this->value || !hasValue){
        this->value = value;
        hasValue = true;
        save(model);
    }
}

void ModelState::saveOnMin(Model &model, double value) {
    if(value < this->value || !hasValue){
        this->value = value;
        hasValue = true;
        save(model);
    }
}
