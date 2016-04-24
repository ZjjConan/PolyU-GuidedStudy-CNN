/*
 */


#include "../CNN/activLayer.h"
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	const int ROWS = 6;
	const int COLS = 6;
	const int CHNS = 2;
	const int NUMIMAGES = 1;

	vector<vector<Mat>> A(NUMIMAGES);
	for (int i = 0; i < NUMIMAGES; ++i) {
		A[i].resize(CHNS);
		for (int j = 0; j < CHNS; ++j) {
			A[i][j] = Mat::zeros(ROWS, COLS, CV_32FC1);
			randn(A[i][j], 0, 1);
		}
	}
	convnet::ActivLayer *node = new convnet::ActivLayer(A);
	node->setActivFuncName("RELU");
	node->init();

	for (int i = 0; i < 100; ++i) {
		double extime = (double)cv::getTickCount();
		node->fprop();
		printf("activ fprop time %fs\n", ((double)cv::getTickCount() - extime) / cv::getTickFrequency());
		extime = (double)cv::getTickCount();
		node->bprop();
		printf("activ bprop time %fs\n", ((double)cv::getTickCount() - extime) / cv::getTickFrequency());
	}

	cout << node->getNFCOuFeatMaps()[0][0] << endl;

	if (node != NULL) delete node; node = NULL;

	return 0;
}