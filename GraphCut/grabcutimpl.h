#ifndef GRAB_CUT_IMPL_H
#define GRAB_CUT_IMPL_H

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>
#include <iostream>
#include <cmath>
#include <cstdio>
#include "maxflow-v3.01/graph.h" //最小割最大流
#include <ctime>

using namespace std;
using namespace cv;

class GMM
{
public:
	static const int componentsCount = 5;

	GMM(int i = 0);
	double operator()(const Vec3d color) const;
	double operator()(int ci, const Vec3d color) const;
	int whichComponent(const Vec3d color) const;

	void initLearning();
	void addSample(int ci, const Vec3d color);
	void endLearning();
	static void grabcutInit(std::string);
	static void grabcutCalc();

private:
	void calcInverseCovAndDeterm(int ci);
	Mat model;
	double* coefs;
	double* mean;
	double* cov;

	double inverseCovs[componentsCount][3][3]; //协方差的逆矩阵
	double covDeterms[componentsCount];  //协方差的行列式

	double sums[componentsCount][3];
	double prods[componentsCount][3][3];
	int sampleCounts[componentsCount];
	int totalSampleCount;

};

#endif