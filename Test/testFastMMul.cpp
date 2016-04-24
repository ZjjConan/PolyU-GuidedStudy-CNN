/*
*/


#include "../Utility/mmul.h"
#include "../Utility/check.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <arrayfire.h>

using namespace cv;
using namespace af;
using namespace std;


int main()
{
	Mat A(2048, 2048, CV_32FC1); cv::randn(A, 0, 1);
	Mat B(2048, 2048, CV_32FC1); cv::randn(B, 0, 1);
	Mat C(2048, 2048, CV_32FC1);
	
	__declspec(align(16)) float *AA = (float *)calloc(2048*2048, sizeof(float));
	__declspec(align(16)) float *BB = (float *)calloc(2048*2048, sizeof(float));
	__declspec(align(16)) float *CC = (float *)calloc(2048*2048, sizeof(float));
	for (int i = 0; i < 2048*2048; ++i)
	{
		AA[i] = 0.1f;
		BB[i] = 0.1f;
	}
	
	for (int i = 0; i < 10; ++i) {
		double extime = (double)cv::getTickCount();
		convnet::fastMatMul(CV_MAT_PRF(C), CV_MAT_PRF(A), CV_MAT_PRF(B),
						    A.rows, A.cols, B.rows, B.cols);
		printf("fastMatMul time %fs\n", ((double)cv::getTickCount() - extime) / cv::getTickFrequency());

		extime = (double)cv::getTickCount();
		convnet::mmul(CC, AA, BB, 2048, 2048, 2048);
		printf("sseMatMul time %fs\n", ((double)cv::getTickCount() - extime) / cv::getTickFrequency());
	}
	

}