/*
*/

#include "../Utility/im2row.h"
#include <opencv2/core/core.hpp>
#include <vector>

using namespace cv;
using namespace std;

static void printMat(cv::Mat &mat)
{
	for (int r = 0; r < mat.rows; ++r) {
		for (int c = 0; c < mat.cols; ++c) {
			printf("%.0f ", mat.at<float>(r, c));
		}
		printf("\n");
	}
	printf("\n");
}


int main()
{
	const int ROWS = 4;
	const int COLS = 4;
	const int CHNS = 2;
	const int NUMIMAGES = 1;
	const int winHeight = 4; 
	const int winWidth = 4;
	const int stepRow = 2;
	const int stepCol = 2;
	const int padTop = 2;
	const int padLeft = 2;
	const int padBottom = 2;
	const int padRight = 2;

	vector<vector<Mat>> A(NUMIMAGES);
	for (int i = 0; i < NUMIMAGES; ++i) {
		A[i].resize(CHNS);
		for (int j = 0; j < CHNS; ++j) {
			A[i][j] = (Mat_<float>(ROWS, COLS) << 1, 1, 2, 3, 1, 5, 5, 7, 8, 1, 1, 2, 3, 1, 5, 5);
		}
	}
	printMat(A[0][0]); printMat(A[0][1]);

	Mat colImage(32, 9, CV_32FC1);
	Mat rowImage(9, 32, CV_32FC1);
	for (int i = 0; i < 1; ++i) {
		convnet::im2col(colImage, A[0], winHeight, winWidth, stepRow, stepCol, 
 						padTop, padLeft, padBottom, padRight);
		convnet::im2row(rowImage, A[0], winHeight, winWidth, stepRow, stepCol,
					    padTop, padLeft, padBottom, padRight);
	}

	printMat(colImage);
	printMat(rowImage);
	Mat diff = colImage.t() - rowImage;
	printMat(diff);
	
	return 0;
}