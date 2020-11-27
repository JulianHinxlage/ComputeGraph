//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "MeanBuffer.h"

MeanBuffer::MeanBuffer(int maxValues) {
    this->maxValues = maxValues;
    index = 0;
    sum = 0;
}

void MeanBuffer::add(double value) {
    if(values.size() < maxValues){
        values.push_back(value);
        sum += value;
    }else{
        index %= values.size();
        sum -= values[index];
        sum += value;
        values[index] = value;
        index++;
    }
}

double MeanBuffer::mean(){
    if(values.size() == 0){
        return 0;
    }else{
        return sum / values.size();
    }
}

void MeanBuffer::clear() {
    sum = 0;
    index = 0;
    values.clear();
}
