#ifndef _CONVNET_UTILITY_IM2ROW_H_
#define _CONVNET_UTILITY_IM2ROW_H_
#pragma once


#include <vector>			     // vector
#include <opencv2/core/core.hpp> // Mat

namespace convnet
{
	using namespace std;
	using namespace cv;
	
	// cv::Mat is row major, not correspondence with Matlab
	// so im2row() convert image into row-major "col" for matrix-matrix operation
	void im2row(Mat &colImage, const vector<Mat> &images,
				const int winHeight, const int winWidth, 
				const int stepRow, const int stepCol,
				const int padTop, const int padLeft,
				const int padBottom, const int padRight);

	void row2im(vector<Mat> &images, const Mat &colImage,
				const int winHeight, const int winWidth,
				const int stepRow, const int stepCol,
				const int padTop, const int padLeft,
				const int padBottom, const int padRight);


	void im2col(Mat &colImage, const vector<Mat> &images,
				const int winHeight, const int winWidth,
				const int stepRow, const int stepCol,
				const int padTop, const int padLeft,
				const int padBottom, const int padRight);

	void col2im(vector<Mat> &images, const Mat &colImage,
				const int winHeight, const int winWidth,
				const int stepRow, const int stepCol,
				const int padTop, const int padLeft,
				const int padBottom, const int padRight);
}




#endif // _convnet_utility_include_im2row_h_