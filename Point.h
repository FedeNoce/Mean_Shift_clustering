//
// Created by federico on 14/03/22.
//

#ifndef MEAN_SHIFT_CLUSTERING_POINT_H
#define MEAN_SHIFT_CLUSTERING_POINT_H

#include <printf.h>

class Point {

public:
    Point(double x_coord, double y_coord){
        this->x_coord = x_coord;
        this->y_coord = y_coord;
    }

    Point(){
        x_coord = 0;
        y_coord = 0;
    }

    double get_x_coord(){
        return this->x_coord;
    }

    double get_y_coord(){
        return this->y_coord;
    }

    void set_x_coord(double x){
        this->x_coord=x;
    }

    void set_y_coord(double y){
        this->y_coord=y;
    }


private:
    double x_coord;
    double y_coord;

};

#endif //MEAN_SHIFT_CLUSTERING_POINT_H
