/*
*/

#include "../CNN/concatLayer.h"
#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main()
{
	const int ROWS = 4;
	const int COLS = 4;
	const int CHNS = 20;
	const int NUMIMAGES = 5;

	vector<vector<Mat>> A(NUMIMAGES);
	for (int i = 0; i < NUMIMAGES; ++i) {
		A[i].resize(CHNS);
		for (int j = 0; j < CHNS; ++j) {
			A[i][j] = Mat::zeros(ROWS, COLS, CV_32FC1);
			randn(A[i][j], 0, 1);
			cout << A[i][j] << endl;
		}
	}
	cout << endl;

	convnet::ConcatLayer *coc = new convnet::ConcatLayer(A);
	coc->setWeightGeometry(COLS, ROWS);
	coc->init();
	for (int i = 0; i < 10; ++i) {
		double extime = (double)cv::getTickCount();
		coc->fprop();
		cout << coc->getFCOuFeatMaps().row(4) << endl;
		printf("conv fprop time %fs\n", ((double)cv::getTickCount() - extime) / cv::getTickFrequency());
		extime = (double)cv::getTickCount();
		coc->bprop();
		printf("conv bprop time %fs\n", ((double)cv::getTickCount() - extime) / cv::getTickFrequency());
	}
	return 0;
}