//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Shape.h"

Shape::Shape() {}

Shape::Shape(std::initializer_list<int> shape) {
    this->dimensions = shape;
}

Shape::Shape(const std::vector<int> &shape) {
    this->dimensions = shape;
}

bool Shape::operator==(const Shape &shape) const {
    return dimensions == shape.dimensions;
}

int Shape::get(int index) const {
    if(index < 0 || index >= rank()){
        return 1;
    }else{
        return dimensions[index];
    }
}

int Shape::elements() const {
    int count = 1;
    for(auto &i : dimensions){
        count *= i;
    }
    return count;
}

int Shape::rank() const{
    return dimensions.size();
}

Tensor Shape::zeros() const {
    return xt::zeros<double>(dimensions);
}
