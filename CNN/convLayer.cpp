/*
*/

#include "../Utility/check.h"
#include "../utility/mmul.h"
#include "../utility/im2row.h"
#include "convLayer.h"

#include <ctime>
#include <arrayfire.h> 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace convnet
{
	using namespace std;
	using namespace cv;
	
	ConvLayer::ConvLayer(Mat4D &inFeatMaps, const int numThreads)
	{
		this->inFeatMaps = inFeatMaps;
		this->numThreads = numThreads;
	}

	ConvLayer::~ConvLayer()
	{
		weights.clear();
		weightMoments.clear();
		weightGrads.clear();
		bias.release();
		biasMoments.release();
		biasGrads.clear();
		inFeatMaps.clear();
		ouFeatMaps.clear();
	}

	void ConvLayer::init()
	{	
		NONFC_INPUT_INIT(inFeatMaps);
		
		int numImages = inFeatMaps.size();
		int weightDims = wparams.height * wparams.width * wparams.weightChns;
		int numWeights = wparams.numWeights;

		// weights, weightMoments
		int numGroups = wparams.numGroups;
		weights.resize(numGroups);
		weightMoments.resize(numGroups);
		for (int gi = 0; gi < numGroups; ++gi) {
			weightMoments[gi] = Mat::zeros(numWeights / numGroups, weightDims, CV_32FC1);
			weights[gi] = Mat::zeros(numWeights / numGroups, weightDims, CV_32FC1);
			
		#ifdef _DEBUG
			srand(0);
		#else
			srand(unsigned int(time(NULL)));
		#endif
			cv::randn(weights[gi], 0, 1);
			if (wparams.initWeightScale > 0)
				weights[gi] *= wparams.initWeightScale;
		}

		// allocate space for bias
		biasMoments = Mat::zeros(numWeights, 1, CV_32FC1);
		bias = Mat::zeros(numWeights, 1, CV_32FC1);
			
		// allocate space for gradients of weights and bias
		weightGrads.resize(numThreads);
		biasGrads.resize(numThreads);
		for (int b = 0; b < numThreads; ++b) {
			biasGrads[b] = Mat::zeros(numWeights, 1, CV_32FC1);
			weightGrads[b].resize(numGroups);
			for (int g = 0; g < numGroups; ++g) {
				weightGrads[b][g] = Mat::zeros(numWeights / numGroups, weightDims, CV_32FC1);
			}
		}

		// allocate space for output maps
		int rows = inFeatMaps[0][0].rows;
		int cols = inFeatMaps[0][0].cols;
		int ouRows = (rows + padding.top + padding.bottom - wparams.height) / 
					  strides.stepRow + 1;
		int ouCols = (cols + padding.left + padding.right - wparams.width) / 
					  strides.stepCol + 1;
		ouFeatMaps.resize(numImages);
		for (int i = 0; i < numImages; i++) {
			ouFeatMaps[i].resize(numWeights);
			for (int j = 0; j < numWeights; j++) {
				ouFeatMaps[i][j] = Mat::zeros(ouRows, ouCols, CV_32FC1);
			}
		}
	}
	
	void ConvLayer::fprop()
	{
		NONFC_INPUT_INIT(inFeatMaps);
		NONFC_OUTPUT_INIT(ouFeatMaps);

		// do fprop
		int numImages = inFeatMaps.size();

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		
		for (int i = 0; i < numImages; ++i) {
			fpropOne(ouFeatMaps[i], inFeatMaps[i], weights, bias,
					 wparams, strides, padding);
		}
		
	}

	void ConvLayer::bprop()
	{
		// must be init() before fprop
		NONFC_INPUT_INIT(inFeatMaps);
		NONFC_OUTPUT_INIT(ouFeatMaps);

		int numImages = ouFeatMaps.size();
		int tsize = numImages < numThreads ? numImages : (int)ceil(numImages / numThreads);
		int t = 0;

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif

		// divide data into number threads (map)
		for (t = 0; t < numThreads; ++t) {
			Mat threadBiasGrads = Mat::zeros(bias.size(), CV_32FC1);
			Mat3D threadWeightGrads(weights.size());
			for (int g = 0; g < wparams.numGroups; ++g) {
				threadWeightGrads[g] = Mat::zeros(weights[0].size(), CV_32FC1);
			}

			int tstart = max(0, t * tsize);
			int tend = min(numImages, (t + 1) * tsize);
			for (int i = tstart; i < tend; ++i) {
				bpropOne(inFeatMaps[i], threadWeightGrads, threadBiasGrads, inFeatMaps[i], ouFeatMaps[i], 
						 weights, wparams, strides, padding, isDzDx);
			}

			for (int g = 0; g < wparams.numGroups; ++g) {
				memcpy(weightGrads[t][g].data, threadWeightGrads[g].data,
					   threadWeightGrads[g].rows * threadWeightGrads[g].cols * sizeof(float));

				memcpy(biasGrads[t].data, threadBiasGrads.data,
					   threadBiasGrads.cols * sizeof(float));
			}
		}

		// accumulate gradients into *[0] across different threads (reduce)
		for (int g = 0; g < wparams.numGroups; ++g) {
			for (int t = 1; t < numThreads; ++t) {
				weightGrads[0][g] += weightGrads[t][g];
			}
		}

		for (int t = 1; t < numThreads; ++t) {
			biasGrads[0] += biasGrads[t];
		}
	}

	void ConvLayer::update()
	{
		layerUpdater.SGDUpdate(weights, weightMoments, bias,
							   biasMoments, weightGrads[0], biasGrads[0]);
	}

	void ConvLayer::scaleLearningRate()
	{
		layerUpdater.scaleLearningRate();
	}


	// ----------------------------------------------------------------------------
	//
	//								private function impl
	//
	// ----------------------------------------------------------------------------	
	void ConvLayer::fpropOne(Mat3D &ouFeatMaps, 
							 const Mat3D &inFeatMaps, 
							 const Mat3D &weights, 
							 const Mat &bias, 
							 const WeightGeometry &wparams, 
							 const StrideGeometry &strides, 
							 const PadGeometry &padding)
	{
		int ouDims = ouFeatMaps[0].rows * ouFeatMaps[0].cols;
		int inDims = inFeatMaps.size() * wparams.height * wparams.width * wparams.numGroups;
		Mat colImage(inDims, ouDims, CV_32FC1);
 		im2col(colImage, inFeatMaps, wparams.height, wparams.width,
     		   strides.stepRow, strides.stepCol, padding.top,
 			   padding.left, padding.bottom, padding.right);

 		int colImageGroupOffset = inDims / wparams.numGroups;
 		int weightGroupOffset = weights[0].rows;
 		Mat ouMap(weightGroupOffset * wparams.numGroups, ouDims, CV_32FC1);
		Mat groupColImage(colImageGroupOffset, colImage.cols, CV_32FC1);
 		Mat groupOuMap(ouDims, weightGroupOffset, CV_32FC1);
		
 		for (int g = 0; g < wparams.numGroups; ++g) {
 			groupColImage = colImage.rowRange(g*colImageGroupOffset, (g + 1)*colImageGroupOffset);
 			groupOuMap = ouMap.rowRange(g*weightGroupOffset, (g + 1)*weightGroupOffset);
    		fastMatMul(CV_MAT_PRF(groupOuMap), CV_MAT_PRF(weights[g]), CV_MAT_PRF(groupColImage),
 				       weights[g].rows, weights[g].cols, groupColImage.rows, groupColImage.cols,
 					   false, false);
 		}
 		
		if (!bias.empty())
			ouMap += repeat(bias, 1, ouMap.cols);

   		// copy to output map, point operation of cv::Mat is faster
		for (int ch = 0; ch < ouMap.rows; ++ch) {
			memcpy(CV_MAT_PRF(ouFeatMaps[ch]), CV_MAT_PRF(ouMap.row(ch)), ouMap.cols * sizeof(float));
		}
	}

	void ConvLayer::bpropOne(Mat3D &prevLayerDelta,
							 Mat3D &weightGrads, 
							 Mat &biasGrads,
							 const Mat3D &inFeatMaps,
							 const Mat3D &currLayerDelta, 
							 const Mat3D &weights, 
							 const WeightGeometry &wparams, 
							 const StrideGeometry &strides,
							 const PadGeometry &padding,
							 const bool isDzDx)
	{
		int numWeights = wparams.numWeights;
		int currDeltaDims = currLayerDelta[0].rows * currLayerDelta[0].cols;
		int inDims = inFeatMaps.size() * wparams.height * wparams.width * wparams.numGroups;
		
		// convert [N x Rows x Cols] delta maps into 2D matrix [N x [Rows x Cols]]
		Mat delta(numWeights, currDeltaDims, CV_32FC1);
		for (int k = 0; k < numWeights; ++k) 
			memcpy(delta.row(k).data, currLayerDelta[k].data, currDeltaDims * sizeof(float));

		// compute bias gradients based on current delta
		// bias [1 x N], delta maps [N x rows x cols] 
		if (!biasGrads.empty()) {
			Mat reduceSum;
			reduce(delta, reduceSum, 1, CV_REDUCE_SUM);
			biasGrads += reduceSum;
		}

		// compute weights gradients based on current delta
		Mat colImage(inDims, currDeltaDims, CV_32FC1);
		im2col(colImage, inFeatMaps, wparams.height, wparams.width, strides.stepRow,
 			   strides.stepCol, padding.top, padding.left, padding.bottom, padding.right);
 		
		int numGroups = weights.size();
		int colImageGroupOffset = inDims / numGroups;
		int weightGroupOffset = numWeights / numGroups;
		Mat groupColImage(colImage.rows, colImageGroupOffset, CV_32FC1);
		Mat groupDeltaMap;
 
 		for (int g = 0; g < numGroups; ++g) {
			groupColImage = colImage.rowRange(g*colImageGroupOffset, (g + 1)*colImageGroupOffset);
 			groupDeltaMap = delta.rowRange(g*weightGroupOffset, (g + 1)*weightGroupOffset);
			fastMatMulAdd(CV_MAT_PRF(weightGrads[g]), CV_MAT_PRF(groupDeltaMap), CV_MAT_PRF(groupColImage),
					      groupDeltaMap.rows, groupDeltaMap.cols, groupColImage.rows, groupColImage.cols, 
						  false, true);
 		}


 		// compute delta(l-1) = dz/dx = kernel * delta(l) 
 		// weights [[chns x wrows x wcols] * N], delta [N x rows x cols]
 		if (isDzDx) {
			int prevDeltaDims = prevLayerDelta[0].rows * prevLayerDelta[0].cols;
			for (int ch = 0; ch < prevLayerDelta.size(); ++ch)
				memset(prevLayerDelta[ch].data, 0, prevDeltaDims * sizeof(float));

 			// should be process to avoid transpose
 			int dzdxGroupOffset = inDims / numGroups;
 			Mat dzdx(inDims, delta.cols, CV_32FC1);
 			Mat groupDzdx, groupWeights;
 
 			for (int g = 0; g < numGroups; ++g) {
 				groupWeights = weights[g];
 				groupDzdx = dzdx.rowRange(g*dzdxGroupOffset, (g + 1)*dzdxGroupOffset);
 				groupDeltaMap = delta.rowRange(g*weightGroupOffset, (g + 1)*weightGroupOffset);
 				fastMatMul(CV_MAT_PRF(groupDzdx), CV_MAT_PRF(groupWeights), CV_MAT_PRF(groupDeltaMap),
 						   groupWeights.rows, groupWeights.cols, groupDeltaMap.rows, groupDeltaMap.cols,
 						   true, false);
 			}
			col2im(prevLayerDelta, dzdx, wparams.height, wparams.width, strides.stepRow,
 				   strides.stepCol, padding.top, padding.left, padding.bottom, padding.right);
 		}
	}
}