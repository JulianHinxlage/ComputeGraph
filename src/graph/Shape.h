//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_SHAPE_H
#define COMPUTEGRAPH_SHAPE_H

#include <vector>
#include "Tensor.h"

class Shape {
public:
    std::vector<int> dimensions;
    Shape();
    Shape(std::initializer_list<int> shape);
    Shape(const std::vector<int> &shape);

    template<typename... T>
    Shape(T... t){
        dimensions = {t...};
    }

    int get(int index) const;

    bool operator==(const Shape &shape) const;

    int elements() const;
    int rank() const;

    Tensor zeros() const;
};


#endif //COMPUTEGRAPH_SHAPE_H
