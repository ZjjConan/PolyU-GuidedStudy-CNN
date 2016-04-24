/*
*/

#include "../Utility/types.h"
#include "../CNN/convLayer.h"
#include "../CNN/poolLayer.h"
#include "../CNN/concatLayer.h"
#include "../CNN/fcLayer.h"
#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


static void printMat(cv::Mat &mat)
{
	for (int r = 0; r < mat.rows; ++r) {
		for (int c = 0; c < mat.cols; ++c) {
			printf("%.6f ", mat.at<float>(r, c));
		}
		printf("\n");
	}
	printf("\n");
}

int main()
{

	const int ROWS = 227;
 	const int COLS = 227;
 	const int CHNS = 3;
 	const int NUMIMAGES = 100;

	vector<vector<Mat>> A(NUMIMAGES);
	for (int i = 0; i < NUMIMAGES; ++i) {
		A[i].resize(CHNS);
		for (int j = 0; j < CHNS; ++j) {
			A[i][j] = Mat::zeros(ROWS, COLS, CV_32FC1);
			randu(A[i][j], 0, 1); 
		}
	}
	//printMat(A[0][0]);
	//printMat(A[0][1]);

	convnet::ConvLayer *node0 = new convnet::ConvLayer(A);
	node0->setWeightGeometry(1, 96, 3, 11, 11, 0.01f);
	node0->setStrideGeometry(4, 4);
	node0->setPadGeometry(0, 0, 0, 0);
	node0->setNumThreads(1);
	node0->setDzDxFlag(false);
  	node0->init();


	for (int i = 0; i < 10; ++i) {
		double extime = (double)cv::getTickCount();
		node0->fprop();
		extime = (double)cv::getTickCount() - extime;
		printf("fprop time %.4fms\n", extime / cv::getTickFrequency() * 1000);

		extime = (double)cv::getTickCount();
		node0->bprop();
		extime = (double)cv::getTickCount() - extime;
		printf("bprop time %.4fms\n", extime / cv::getTickFrequency() * 1000);
 	}
	printf("done !\n");
 	return 0;
}