#ifndef GRAPH_CUT_IMPL_H
#define GRAPH_CUT_IMPL_H

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp> 
#include <opencv2\imgproc\types_c.h>
#include <iostream>
#include <cmath>
#include <cstdio>
#include "maxflow-v3.01/graph.h"

using namespace std;
using namespace cv;

void add2histogram(int col, int row, int& sum, double histogram[256], int OorB);
static void onmouse(int event, int x, int y, int flag, void* param);
double Distance(Vec3d a, Vec3d b);
void graph_cut_work();
double calculate_Beta(Mat& image);
void graph_cut_init(std::string);


#endif
