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

Matrix Shape::zeros() {
    Matrix m;
    int r = 1;
    int c = 1;
    if(dimensions.size() >= 1){
        r = dimensions[0];
    }
    if(dimensions.size() >= 2){
        c = dimensions[1];
    }
    m.setZero(r, c);
    return m;
}
