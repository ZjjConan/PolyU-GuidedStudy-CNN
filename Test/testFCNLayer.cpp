/*
*/

#include "../CNN/fcnLayer.h"
#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main()
{
	const int ROWS = 6;
	const int COLS = 6;
	const int CHNS = 256;
	const int NUMIMAGES = 100;
	
	Mat A(NUMIMAGES, ROWS * COLS * CHNS, CV_32FC1);
	randn(A, 0, 1);

	convnet::FCNLayer *node = new convnet::FCNLayer(A);
	node->setWeightGeometry(1, 4096, 9216, 1, 1, 0.01);
	node->setDzDxFlag(true);
	node->init();
	for (int i = 0; i < 10; ++i) {
		double extime = (double)cv::getTickCount();
		node->fprop();
		printf("conv fprop time %fs\n", ((double)cv::getTickCount() - extime) / cv::getTickFrequency());
		extime = (double)cv::getTickCount();
		node->bprop();
		printf("conv bprop time %fs\n", ((double)cv::getTickCount() - extime) / cv::getTickFrequency());
	}
}