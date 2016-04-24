/*
*/

#include "../Utility/check.h"
#include "../CNN/activLayer.h"

#include <vector>
#include <opencv2/core/core.hpp>        // cv::Mat
#include <opencv2/imgproc/imgproc.hpp>  // cv::threshold
#include <opencv2/imgproc/types_c.h>    // cv::CV_THRESH_BINARY

#ifdef _OPENMP
#include <omp.h>
#endif


namespace convnet
{
	ActivLayer::ActivLayer(Mat4D &inFeatMaps, const int numThreads)
	{
		this->inFeatMaps = inFeatMaps;
		this->numThreads = numThreads;
	}

	ActivLayer::~ActivLayer()
	{
		inFeatMaps.clear();
		tmFeatMaps.clear();
		ouFeatMaps.clear();
		if (activFunc != NULL) {
			delete activFunc;
			activFunc = NULL;
		}
	}

	void ActivLayer::init()
	{
		NONFC_INPUT_INIT(inFeatMaps);

		// allocate space for output feature maps
		int numImages = inFeatMaps.size();
		int chns = inFeatMaps[0].size();
		int rows = inFeatMaps[0][0].rows;
		int cols = inFeatMaps[0][0].cols;
		ouFeatMaps.resize(numImages);
		tmFeatMaps.resize(numImages);
		for (int i = 0; i < numImages; ++i) {
			ouFeatMaps[i].resize(chns);
			tmFeatMaps[i].resize(chns);
			for (int ch = 0; ch < chns; ++ch) {
				ouFeatMaps[i][ch] = Mat::zeros(rows, cols, CV_32FC1);
				tmFeatMaps[i][ch] = Mat::zeros(rows, cols, CV_32FC1);
			}
		}

		// get activation function
		activFunc = getActivFunction(activFuncName);
	}

	void ActivLayer::fprop()
	{
		NONFC_INPUT_INIT(inFeatMaps);
		NONFC_OUTPUT_INIT(ouFeatMaps);

		int numImages = inFeatMaps.size();

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif

		for (int i = 0; i < numImages; ++i) {
			fpropOne(tmFeatMaps[i], inFeatMaps[i], activFunc);
			copyToOutputMaps(ouFeatMaps[i], tmFeatMaps[i]);
		}
	}

	void ActivLayer::bprop()
	{
		NONFC_INPUT_INIT(inFeatMaps);
		NONFC_OUTPUT_INIT(ouFeatMaps);

		int numImages = inFeatMaps.size();

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif

		for (int i = 0; i < numImages; ++i) {
			bpropOne(inFeatMaps[i], tmFeatMaps[i], ouFeatMaps[i],
				     activFunc);
		}
	}



	// ----------------------------------------------------------------------------
	//
	//								private function impl
	//
	// ----------------------------------------------------------------------------
	void ActivLayer::fpropOne(Mat3D &tmFeatMaps, const Mat3D &inFeatMaps,
							  ActivFunction *func)
	{
		for (int ch = 0; ch < inFeatMaps.size(); ++ch) {
			func->fpropOne(tmFeatMaps[ch], inFeatMaps[ch]);
		}
	}

	void ActivLayer::bpropOne(Mat3D &inFeatMaps, const Mat3D &tmFeatMaps,
							  const Mat3D &ouFeatMaps, ActivFunction *func)
	{
		for (int ch = 0; ch < inFeatMaps.size(); ++ch) {
			func->bpropOne(inFeatMaps[ch], tmFeatMaps[ch], ouFeatMaps[ch]);
		}
	}


	void ActivLayer::copyToOutputMaps(Mat3D &ouFeatMaps, const Mat3D &tmFeatMaps)
	{
		int rows = ouFeatMaps[0].rows;
		int cols = ouFeatMaps[0].cols;
		for (int ch = 0; ch < ouFeatMaps.size(); ++ch)
			memcpy(ouFeatMaps[ch].data, tmFeatMaps[ch].data, rows * cols * sizeof(float));
	}
}



namespace convnet
{
	FCActivLayer::FCActivLayer(Mat &inFeatMaps, const int numThreads)
	{
		this->inFeatMaps = inFeatMaps;
		this->numThreads = numThreads;
	}

	FCActivLayer::~FCActivLayer()
	{
		inFeatMaps.release();
		tmFeatMaps.release();
		ouFeatMaps.release();
		if (activFunc != NULL) {
			delete activFunc;
			activFunc = NULL;
		}
	}


	void FCActivLayer::init()
	{
		FC_INPUT_INIT(inFeatMaps);

		// allocate space for output feature maps
		tmFeatMaps = Mat::zeros(inFeatMaps.size(), CV_32FC1);
		ouFeatMaps = Mat::zeros(inFeatMaps.size(), CV_32FC1);

		// get activation function
		activFunc = getActivFunction(activFuncName);
	}

	void FCActivLayer::fprop()
	{
		FC_INPUT_INIT(inFeatMaps);
		FC_OUTPUT_INIT(tmFeatMaps);
		FC_OUTPUT_INIT(ouFeatMaps);

		fpropOne(tmFeatMaps, inFeatMaps, activFunc);
		copyToOutputMaps(ouFeatMaps, tmFeatMaps);
	}

	void FCActivLayer::bprop()
	{
		FC_INPUT_INIT(inFeatMaps);
		FC_OUTPUT_INIT(tmFeatMaps);
		FC_OUTPUT_INIT(ouFeatMaps);

		bpropOne(inFeatMaps, tmFeatMaps, ouFeatMaps, activFunc);
	}

	// ----------------------------------------------------------------------------
	//
	//								private function impl
	//
	// ----------------------------------------------------------------------------
	void FCActivLayer::fpropOne(Mat &tmFeatMaps, const Mat &inFeatMaps,
		                        ActivFunction *func)
	{
		func->fpropOne(tmFeatMaps, inFeatMaps);
	}

	void FCActivLayer::bpropOne(Mat &inFeatMaps, const Mat &tmFeatMaps,
						        const Mat &ouFeatMaps, ActivFunction *func)
	{
		func->bpropOne(inFeatMaps, tmFeatMaps, ouFeatMaps);
	}


	void FCActivLayer::copyToOutputMaps(Mat &ouFeatMaps, const Mat &tmFeatMaps)
	{
		int rows = ouFeatMaps.rows;
		int cols = ouFeatMaps.cols;
		memcpy(ouFeatMaps.data, tmFeatMaps.data, rows * cols * sizeof(float));
	}
}
