/* *
 *
 */

#include "../Utility/check.h"
#include "../Utility/mmul.h"
#include "loss.h"

#include <ctime>
#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

namespace convnet
{

	// ----------------------------------------------------------------------------
	//
	//								Softmax loss
	//
	// ----------------------------------------------------------------------------
	
	SoftmaxLoss::SoftmaxLoss(Mat &inFeatMaps, const Mat &labels, const int numThreads)
	{
		this->inFeatMaps = inFeatMaps;
		this->labels = labels;
		this->numThreads = numThreads;
	}

	SoftmaxLoss::~SoftmaxLoss()
	{
		inFeatMaps.release();
		labels.release();
		lmat.release();
	}

	void SoftmaxLoss::init()
	{
		lmat = Mat::zeros(inFeatMaps.size(), CV_32FC1);
	}
	
	void SoftmaxLoss::fprop()
	{
		memset(lmat.data, 0, lmat.rows * lmat.cols * sizeof(float));
		label2matrix(lmat, labels);

		fpropOne(inFeatMaps);
	}

	void SoftmaxLoss::bprop()
	{
		bpropOne(inFeatMaps, inFeatMaps, lmat);
	}


	// ----------------------------------------------------------------------------
	//
	//								private function impl
	//
	// ----------------------------------------------------------------------------
	void SoftmaxLoss::label2matrix(Mat &lmat, const Mat &label)
	{
		int numData = lmat.rows;
		float *lablPtr = CV_MAT_PRF(label);
		float *lmatPtr = CV_MAT_PRF(lmat);
		for (int i = 0; i < numData; ++i)  {
			int index = *lablPtr++;
			CV_MAT_AT(lmatPtr, i, index, lmat.cols) = 1;
		}
		lablPtr = NULL;
		lmatPtr = NULL;
	}

	void SoftmaxLoss::fpropOne(Mat &inFeatMaps)
	{
		// compute softmax probability
		Mat reduceVal;
		reduce(inFeatMaps, reduceVal, 1, REDUCE_MAX, inFeatMaps.type());
		inFeatMaps -= repeat(reduceVal, 1, inFeatMaps.cols);
		exp(inFeatMaps, inFeatMaps);
		reduce(inFeatMaps, reduceVal, 1, REDUCE_SUM, inFeatMaps.type());
		inFeatMaps = inFeatMaps / repeat(reduceVal, 1, inFeatMaps.cols);
	}

	void SoftmaxLoss::bpropOne(Mat &delta, const Mat &inFeatMaps, const Mat &lmat)
	{
		delta = (inFeatMaps - lmat) / inFeatMaps.rows;
	}
}