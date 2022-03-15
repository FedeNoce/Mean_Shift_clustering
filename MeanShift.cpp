/*
 ============================================================================
 Name        : MeanShiftOpenMp.cpp
 Author      : Federico Nocentini & Corso Vignoli
 Version     :
 Copyright   :
 Description : OpenMp implementation of Mean Shift clustering algorithm
 ============================================================================
 */

#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include "Point.h"
#include <omp.h>
#include <string>
#include <vector>
#include <c++/8/sstream>


using namespace std;
using namespace std::chrono;

int num_point = 1000;
int max_iterations = 20;
int bandwidth = 2;

vector<Point> init_point();
void mean_shift(vector<Point> &points, vector<Point> &shifted_points);


int main() {

    printf("Number of points %d\n", num_point);


    srand(int(time(NULL)));


    vector<Point> points;
    points = init_point();
    vector<Point> shifted_points;
    shifted_points = init_point();


    double time_point1 = omp_get_wtime();


    bool conv = true;
    int iterations = 0;

    printf("Starting iterate...\n");

    //The algorithm stops when iterations > max_iteration or when the clusters didn't move
    while(iterations < max_iterations){

        iterations ++;

        mean_shift(points, shifted_points);

        printf("Iteration %d done \n", iterations);

    }

    double time_point2 = omp_get_wtime();
    double duration = time_point2 - time_point1;

    printf("Number of iterations: %d, total time: %f seconds, time per iteration: %f seconds\n",
           iterations, duration, duration/iterations);

    for(int i=0;i<num_point;++i){
        if(shifted_points[i].get_x_coord() != 0.0){
            float current_centroid_x=shifted_points[i].get_x_coord();
            //float current_centroid_y=h_shifted_datapoints_y[i];
            for(int j=i+1;j<num_point;j++){
                if(abs(shifted_points[j].get_x_coord() - current_centroid_x) < 0.0001){
                    shifted_points[j].set_x_coord(0.0);
                    //h_shifted_datapoints_y[j]=0.0;
                }
            }
        }
    }
    printf("Centroids: \n");
    for(int i=0;i<num_point;++i){
        if(shifted_points[i].get_x_coord()!= 0.0) {
            printf("(%f, %f) \n", shifted_points[i].get_x_coord(), shifted_points[i].get_y_coord());
        }
    }

    return 0;


}

//Initialize num_point Points
vector<Point> init_point(){


    string fname = "/home/federico/CLionProjects/Mean_Shift_clustering/datasets/2D_data_3.csv";
    vector<vector<string>> content;
    vector<string> row;
    string line, word;
    fstream file (fname, ios::in);
    if(file.is_open())
    {
        while(getline(file, line))
        {
            row.clear();

            stringstream str(line);

            while(getline(str, word, ','))
                row.push_back(word);
            content.push_back(row);
        }
    }
    vector<Point> points(num_point);
    Point *ptr = &points[0];
    cout<<"Datapoints:"<<"\n";
    for(int i=0;i<content.size();i++)
    {
        cout<<content[i][0]<<","<<content[i][0]<<"\n";
        Point* point = new Point(std::stod(content[i][0]), std::stod(content[i][1]));
        ptr[i] = *point;
    }
    return points;

}

void mean_shift(vector<Point> &points, vector<Point> &shifted_points){

    unsigned long points_size = points.size();


    for (int i = 0; i < points_size; i++) {
        float diff_x = 0.0;
        float diff_y = 0.0;
        float new_x = 0.0;
        float new_y = 0.0;
        float tot_weight = 0.0;

        for (int j = 0; j < points_size; j++) {
            diff_x = shifted_points[i].get_x_coord() - points[j].get_x_coord();
            diff_y = shifted_points[i].get_y_coord() - points[j].get_y_coord();
            float square_distance = pow(diff_x, 2) + pow(diff_y, 2);
            float weight = exp((-square_distance) / (2 * pow(bandwidth, 2)));
            new_x += points[j].get_x_coord() * weight;
            new_y += points[j].get_y_coord() * weight;
            tot_weight += weight;

        }
        new_x /= tot_weight;
        new_y /= tot_weight;
        shifted_points[i].set_x_coord(new_x);
        shifted_points[i].set_y_coord(new_y);

    }
}



