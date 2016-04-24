/*
*/

#include "concatLayer.h"
#include "../Utility/check.h"
#include "../Utility/im2row.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace convnet 
{
	ConcatLayer::ConcatLayer(Mat4D &inFeatMaps, const int numThreads)
	{
		this->inFeatMaps = inFeatMaps;
		this->numThreads = numThreads;
	}

	ConcatLayer::~ConcatLayer()
	{
		inFeatMaps.clear();
	}

	void ConcatLayer::init()
	{
		NONFC_INPUT_INIT(inFeatMaps);

		int numImages = inFeatMaps.size();
		int dims = inFeatMaps[0].size() * wparams.height * wparams.width;
		int numBlocks = (inFeatMaps[0][0].rows - wparams.height + 1) * 
						(inFeatMaps[0][0].cols - wparams.width + 1);
		ouFeatMaps = Mat::zeros(numBlocks * numImages, dims, CV_32FC1);
	}

	void ConcatLayer::fprop()
	{
		NONFC_INPUT_INIT(inFeatMaps);
		FC_OUTPUT_INIT(ouFeatMaps);

		int numImages = inFeatMaps.size();
		int numBlocks = (inFeatMaps[0][0].rows - wparams.height + 1) *
						(inFeatMaps[0][0].cols - wparams.width + 1);

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif

		for (int i = 0; i < numImages; ++i) {
			fpropOne(ouFeatMaps.rowRange(i, i+numBlocks), inFeatMaps[i],
				     wparams);
		}
	}

	void ConcatLayer::bprop()
	{
		NONFC_INPUT_INIT(inFeatMaps);
		FC_OUTPUT_INIT(ouFeatMaps);

		int numImages = inFeatMaps.size();
		int numBlocks = (inFeatMaps[0][0].rows - wparams.height + 1) *
						(inFeatMaps[0][0].cols - wparams.width + 1);
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif

		for (int i = 0; i < numImages; ++i) {
			bpropOne(inFeatMaps[i], ouFeatMaps.rowRange(i, i+numBlocks),
				     wparams);
		}
	}

	void ConcatLayer::fpropOne(Mat &ouFeatMaps, const Mat3D &inFeatMaps, 
						       const WeightGeometry &wparams)
	{
// 		im2row(ouFeatMaps, inFeatMaps, wparams.height, wparams.width, 
// 			   1, 1, 0, 0, 0, 0);

		int chns = inFeatMaps.size();
		int dimsPerChn = inFeatMaps[0].rows * inFeatMaps[0].cols;
		float *ouMapPtr = CV_MAT_PRF(ouFeatMaps);
		for (int ch = 0; ch < chns; ++ch) {
			memcpy(ouMapPtr + ch * dimsPerChn, CV_MAT_PRF(inFeatMaps[ch]), 
				   dimsPerChn * sizeof(float));
		}
	}

	void ConcatLayer::bpropOne(Mat3D &inFeatMaps, const Mat &ouFeatMaps,
							   const WeightGeometry &wparams)
	{

// 		row2im(inFeatMaps, ouFeatMaps, wparams.height, wparams.width, 
// 			   1, 1, 0, 0, 0, 0);
		int chns = inFeatMaps.size();
		int dimsPerChn = inFeatMaps[0].rows * inFeatMaps[0].cols;
		float *ouMapPtr = CV_MAT_PRF(ouFeatMaps);
		for (int ch = 0; ch < chns; ++ch) {
			memcpy(CV_MAT_PRF(inFeatMaps[ch]), ouMapPtr + ch * dimsPerChn, 
				   dimsPerChn * sizeof(float));
		}
	}
}