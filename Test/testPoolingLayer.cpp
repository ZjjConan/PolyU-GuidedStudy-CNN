/*
*/

#include "../Utility/types.h"
#include "../Utility/param.h"
#include "../CNN/poolLayer.h"

#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

int main()
{
	const int ROWS = 55;
	const int COLS = 55;
	const int CHNS = 96;
	const int NUMIMAGES = 100;

	RNG rng;
	Mat4D A(NUMIMAGES);
	for (int k = 0; k < NUMIMAGES; ++k) {
		A[k].resize(CHNS);
		for (int i = 0; i < CHNS; ++i) {
			A[k][i] = Mat::zeros(ROWS, COLS, CV_32FC1);
			randu(A[k][i], 0, 1);
		}
	}
	
	convnet::PoolLayer *node = new convnet::PoolLayer(A);
	node->setWeightGeometry(3, 3);
	node->setPadGeometry(0, 0, 0, 0);
	node->setStrideGeometry(2, 2);
	node->setPoolMethod("Max");
	node->setScaleMapFlag(false);
	node->init();
	for (int i = 0; i < 10; ++i) {
		double extime = (double)cv::getTickCount();
		node->fprop();
		printf("max pool time %fs\n", ((double)cv::getTickCount() - extime) / (double)cv::getTickFrequency());
 		extime = (double)cv::getTickCount();
 		node->bprop();
 		printf("max unpool time %fs\n", ((double)cv::getTickCount() - extime) / (double)cv::getTickFrequency());
	}
	free(node); node = NULL;
	return 0;
}