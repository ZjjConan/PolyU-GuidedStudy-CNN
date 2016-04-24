#ifndef _CONVNET_IMDB_CIFAR_H_
#define _CONVNET_IMDB_CIFAR_H_
#pragma once

#include <cstdio>   // fopen
#include <cstdlib>  // abort
#include <cstring>  // string
#include <cassert>  // assert
#include <iostream> // cout
#include <cmath>    // sqrt 
#include <ctime>    // time

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

namespace imdb
{
	using namespace std;
	using namespace cv;

	class CIFAR10
	{
	public:
		CIFAR10() {}
		~CIFAR10() {}

		void loadBatch(vector<vector<Mat>> &images, Mat &labels, const string &fileName);

		void showImages(vector<vector<Mat>> &images, const int numShowed = 100);
	};

	void CIFAR10::loadBatch(vector<vector<Mat>> &images, Mat &labels, const string &fileName)
	{
		const int numImages = 10000;
		const int rows = 32;
		const int cols = 32;
		const int chns = 3;

		assert(fileName.empty() == false);

		FILE *file = NULL;
		errno_t err = fopen_s(&file, fileName.c_str(), "rb");
		//FILE *file = fopen(fileName.c_str(), "rb");
		if (err != 0) {
			printf("Could not open %s\n", fileName.c_str());
			abort();
		}

		images.resize(numImages);
		for (int i = 0; i < numImages; ++i) {
			images[i].resize(chns);
			for (int ch = 0; ch < chns; ++ch) {
				images[i][ch] = Mat::zeros(rows, cols, CV_8UC1);
			}
		}

		labels = Mat::zeros(1, numImages, CV_8UC1);
		uchar *labelPtr = (uchar *)labels.data;
		for (int i = 0; i < numImages; ++i) {
			fread(&labelPtr[i], sizeof(uchar), 1, file);
			fread(images[i][0].data, sizeof(uchar), rows*cols, file);
			fread(images[i][1].data, sizeof(uchar), rows*cols, file);
			fread(images[i][2].data, sizeof(uchar), rows*cols, file);
		}
		labels.convertTo(labels, CV_32FC1);
		labelPtr = NULL;
		fclose(file);
	}

	void CIFAR10::showImages(vector<vector<Mat>> &images, const int numShowed)
	{
		const int rows = 32;
		const int cols = 32;

		vector<int> index(images.size(), CV_8UC1);
		for (size_t i = 0; i < images.size(); ++i) {
			index[i] = i;
		}
		srand((uchar)time(NULL));
		std::random_shuffle(index.begin(), index.end());

		int sr = 0;
		int sc = -1;
		int numPerRow = (int)ceil(sqrt(numShowed));
		Mat showedImage(numPerRow * rows, numPerRow * cols, CV_8UC3);
		Mat temp(rows, cols, CV_8UC3);
		for (int i = 0; i < numShowed; i++) {
			sr = i % numPerRow;
			if (sr == 0) sc += 1;
			cv::merge(images[index[i]], temp);
			temp.copyTo(showedImage(Range(sr * rows, (sr + 1) * cols), Range(sc * rows, (sc + 1) * cols)));
		}
		imshow("random images", showedImage);
		waitKey(0);

		index.clear();
		showedImage.release();
	}
}

#endif // cifar dataset