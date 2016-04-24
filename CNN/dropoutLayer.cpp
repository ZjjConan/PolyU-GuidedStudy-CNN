/*
*/

#include "../Utility/check.h"
#include "dropoutLayer.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

namespace convnet
{
	DropoutLayer::DropoutLayer(Mat4D &inFeatMaps, const int numThreads)
	{
		this->inFeatMaps = inFeatMaps;
		this->numThreads = numThreads;
	}

	DropoutLayer::~DropoutLayer()
	{
		inFeatMaps.clear();
		ouFeatMaps.clear();
		mask.clear();
	}

	void DropoutLayer::init()
	{
		NONFC_INPUT_INIT(inFeatMaps);

		// allocate space for output maps
		int numImages = inFeatMaps.size();	
		int ouChns = inFeatMaps[0].size();
		int ouRows = inFeatMaps[0][0].rows;
		int ouCols = inFeatMaps[0][0].cols;
		ouFeatMaps.resize(numImages);
		for (int i = 0; i < numImages; i++) {
			ouFeatMaps[i].resize(ouChns);
			for (int j = 0; j < ouChns; j++) {
				ouFeatMaps[i][j] = Mat::zeros(ouRows, ouCols, CV_32FC1);
			}
		}

		if (!isStaticMask) {
			mask.resize(numImages);
			for (int i = 0; i < numImages; i++) {
				mask[i].resize(ouChns);
				for (int j = 0; j < ouChns; j++) {
					mask[i][j] = Mat::zeros(ouRows, ouCols, CV_32FC1);
				}
			}
		}
	}

	void DropoutLayer::fprop()
	{
		NONFC_INPUT_INIT(inFeatMaps);

		int numImages = inFeatMaps.size();

		// pre-create mask for speed-up
		float scale = 1 / (1 - dropoutRate);

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif

		for (int i = 0; i < numImages; ++i) {
			for (int ch = 0; ch < inFeatMaps[0].size(); ++ch) {
				cv::randu(mask[i][ch], 0, 1);
				cv::threshold(mask[i][ch], mask[i][ch], dropoutRate, scale, CV_THRESH_BINARY);
			}
		}

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif

		for (int i = 0; i < numImages; i++) {
			fpropOne(ouFeatMaps[i], mask[i], inFeatMaps[i], isStaticMask);
		}
	}


	void DropoutLayer::bprop()
	{
		NONFC_INPUT_INIT(inFeatMaps);

		int numImages = inFeatMaps.size();

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif

		for (int i = 0; i < numImages; i++) {
			bpropOne(inFeatMaps[i], ouFeatMaps[i], mask[i]);
		}
	}


	// ----------------------------------------------------------------------------
	//
	//								private function impl
	//
	// ----------------------------------------------------------------------------
	void DropoutLayer::fpropOne(Mat3D &ouFeatMaps, 
							    const Mat3D &mask,
						        const Mat3D &inFeatMaps,
							    const bool isStaticMask)
	{
		if (!isStaticMask) {
			for (int ch = 0; ch < inFeatMaps.size(); ++ch) {
				ouFeatMaps[ch] = inFeatMaps[ch].mul(mask[ch]);
			}
		}
		else
			for (int ch = 0; ch < inFeatMaps.size(); ++ch)
				ouFeatMaps[ch] = inFeatMaps[ch].mul(mask[ch]);
	}

	void DropoutLayer::bpropOne(Mat3D &inFeatMaps, const Mat3D &ouFeatMaps, 
							    const Mat3D &mask)
	{
		for (int ch = 0; ch < ouFeatMaps.size(); ++ch) {
			inFeatMaps[ch] = ouFeatMaps[ch].mul(mask[ch]);
		}
	}
}


namespace convnet
{
	FCDropoutLayer::FCDropoutLayer(Mat &inFeatMaps, const int numThreads)
	{
		this->inFeatMaps = inFeatMaps;
		this->numThreads = numThreads;
	}

	FCDropoutLayer::~FCDropoutLayer()
	{
		inFeatMaps.release();
		ouFeatMaps.release();
		mask.release();
	}

	void FCDropoutLayer::init()
	{
		FC_INPUT_INIT(inFeatMaps);

		// allocate space for output maps
		ouFeatMaps = Mat::zeros(inFeatMaps.size(), CV_32FC1);

		if (!isStaticMask) {
			mask = Mat::zeros(inFeatMaps.size(), CV_32FC1);
		}
	}

	void FCDropoutLayer::fprop()
	{
		FC_INPUT_INIT(inFeatMaps);

		fpropOne(ouFeatMaps, mask, inFeatMaps, dropoutRate, isStaticMask);
	}


	void FCDropoutLayer::bprop()
	{
		FC_INPUT_INIT(inFeatMaps);

		bpropOne(inFeatMaps, ouFeatMaps, mask);
	}


	// ----------------------------------------------------------------------------
	//
	//								private function impl
	//
	// ----------------------------------------------------------------------------
	void FCDropoutLayer::fpropOne(Mat &ouFeatMaps, Mat &mask, 
								  const Mat &inFeatMaps, 
								  const float dropoutRate,
								  const bool isStaticMask)
	{
		float scale = 1 / (1 - dropoutRate);
		if (!isStaticMask) {
			cv::randu(mask, 0, 1);
			cv::threshold(mask, mask, dropoutRate, scale, CV_THRESH_BINARY);
			ouFeatMaps = inFeatMaps.mul(mask);
		}
		else
			ouFeatMaps = inFeatMaps.mul(mask);
	}

	void FCDropoutLayer::bpropOne(Mat &inFeatMaps, const Mat &ouFeatMaps,
							      const Mat &mask)
	{
		inFeatMaps = ouFeatMaps.mul(mask);
	}
}