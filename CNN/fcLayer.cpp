/*
*/

#include "../Utility/check.h"
#include "../Utility/mmul.h"
#include "fcLayer.h"
#include <ctime>

namespace convnet
{
	FCLayer::FCLayer(Mat &inFeatMaps, const int numThreads)
	{
		this->inFeatMaps = inFeatMaps;
		this->numThreads = numThreads;
	}

	FCLayer::~FCLayer()
	{
		inFeatMaps.release();
		ouFeatMaps.release();
	}

	void FCLayer::init()
	{
		FC_INPUT_INIT(inFeatMaps);

		int numImages = inFeatMaps.rows;
		int weightDims = wparams.height * wparams.width * wparams.weightChns;
		int numWeights = wparams.numWeights;

		// weights, weightMoments
		weightMoments = Mat::zeros(weightDims, numWeights, CV_32FC1);
		weights = Mat::zeros(weightDims, numWeights, CV_32FC1);
		
	#if _DEBUG
		srand(0);
	#else
		srand(unsigned int(time(NULL)));
	#endif
		cv::randn(weights, 0, 1);
		if (wparams.initWeightScale > 0)
			weights *= wparams.initWeightScale;

		// allocate space for bias
		biasMoments = Mat::zeros(1, numWeights, CV_32FC1);
		bias = Mat::zeros(1, numWeights, CV_32FC1);

		// allocate space for gradients of weights and bias
		biasGrads = Mat::zeros(1, numWeights, CV_32FC1);
		weightGrads = Mat::zeros(weightDims, numWeights, CV_32FC1);

		// allocate space for output maps
		ouFeatMaps = Mat::zeros(inFeatMaps.rows, numWeights, CV_32FC1);
	}

	void FCLayer::fprop()
	{
		FC_INPUT_INIT(inFeatMaps);
		FC_OUTPUT_INIT(ouFeatMaps);

		fpropOne(ouFeatMaps, inFeatMaps, weights, bias);
	}
	

	void FCLayer::bprop()
	{
		FC_INPUT_INIT(inFeatMaps);
		FC_OUTPUT_INIT(ouFeatMaps);

		bpropOne(inFeatMaps, weightGrads, biasGrads, 
				 inFeatMaps, ouFeatMaps, weights, isDzDx);
	}

	void FCLayer::update()
	{
		layerUpdater.SGDUpdate(weights, weightMoments, bias, 
							   biasMoments, weightGrads, biasGrads);
	}

	void FCLayer::scaleLearningRate()
	{
		layerUpdater.scaleLearningRate();
	}


	// ----------------------------------------------------------------------------
	//
	//								private function impl
	//
	// ----------------------------------------------------------------------------
	void FCLayer::fpropOne(Mat &ouFeatMaps, const Mat &inFeatMaps,
						   const Mat &weights, const Mat &bias)
	{
		fastMatMul(CV_MAT_PRF(ouFeatMaps), CV_MAT_PRF(inFeatMaps), CV_MAT_PRF(weights),
				   inFeatMaps.rows, inFeatMaps.cols, weights.rows, weights.cols,
				   false, false);

		if (!bias.empty())
			ouFeatMaps += repeat(bias, ouFeatMaps.rows, 1);
	}

	void FCLayer::bpropOne(Mat &prevLayerDelta,
						   Mat &weightGrads, 
						   Mat &biasGrads, 
						   const Mat &inFeatMaps, 
						   const Mat &currLayerDelta, 
						   const Mat &weights, 
						   const bool isDzDx)
	{
		// compute bias gradients based on current delta
		// bias [1 x numWeights], delta maps [numImages x numWeights] 
		if (!biasGrads.empty()) {
			reduce(currLayerDelta, biasGrads, 0, REDUCE_SUM, CV_32FC1);
		}
		

		// compute weights gradients based on current delta
		fastMatMul(CV_MAT_PRF(weightGrads), CV_MAT_PRF(inFeatMaps), CV_MAT_PRF(currLayerDelta),
				   inFeatMaps.rows, inFeatMaps.cols, currLayerDelta.rows, currLayerDelta.cols,
				   true, false);

		// compute delta(l-1) = dz/dx = delta(l) * weight^T 
		if (isDzDx) {
			fastMatMul(CV_MAT_PRF(prevLayerDelta), CV_MAT_PRF(currLayerDelta), CV_MAT_PRF(weights),
					   currLayerDelta.rows, currLayerDelta.cols, weights.rows, weights.cols,
				       false, true);
		}
	}
}