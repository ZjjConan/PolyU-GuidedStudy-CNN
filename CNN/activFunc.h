#ifndef _CONVNET_CNN_ACTIVFUNC_H_
#define _CONVNET_CNN_ACTIVFUNC_H_
#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace convnet
{
	using namespace cv;

	// activation function interface
	class ActivFunction
	{
	public:
		virtual ~ActivFunction() {}

		virtual void fpropOne(Mat &actFeatMaps, const Mat &inFeatMaps) = 0;

		virtual void bpropOne(Mat &prevLayerDelta, const Mat &acFeatMaps,
							  const Mat &currLayerDelta) = 0;
	};
}


namespace convnet
{
	using namespace std;
	using namespace cv;

	// ----------------------------------------------------------------------------
	//
	//							linear activation function
	//
	// ----------------------------------------------------------------------------
	class LinearFunction : public ActivFunction
	{
	public:
		LinearFunction() {}
		virtual ~LinearFunction() {}

		virtual void fpropOne(Mat &acFeatMaps, const Mat &inFeatMaps)
		{
			int rows = inFeatMaps.rows;
			int cols = inFeatMaps.cols;
			memcpy(acFeatMaps.data, inFeatMaps.data, rows * cols * sizeof(float));
		}

		virtual void bpropOne(Mat &prevLayerDelta, const Mat &acFeatMaps,
							  const Mat &currLayerDelta)
		{
			int rows = currLayerDelta.rows;
			int cols = currLayerDelta.cols;
			memcpy(prevLayerDelta.data, currLayerDelta.data, rows * cols * sizeof(float));
		}
	};


	// ----------------------------------------------------------------------------
	//
	//							sigmoid activation function
	//
	// ----------------------------------------------------------------------------
	class SigmoidFunction : public ActivFunction
	{
	public:
		SigmoidFunction() {}
		virtual ~SigmoidFunction() {}

		virtual void fpropOne(Mat &acFeatMaps, const Mat &inFeatMaps)
		{
			cv::exp(-inFeatMaps, acFeatMaps);
			acFeatMaps = 1 / (acFeatMaps + 1);
		}

		virtual void bpropOne(Mat &prevLayerDelta, const Mat &acFeatMaps,
							  const Mat &currLayerDelta)
		{
			prevLayerDelta = acFeatMaps.mul(1 - acFeatMaps).mul(currLayerDelta);
		}
	};


	// ----------------------------------------------------------------------------
	//
	//							  relu activation function
	//
	// ----------------------------------------------------------------------------
	class ReLUFunction : public ActivFunction
	{
	public:
		ReLUFunction() {}
		virtual ~ReLUFunction() {}

		virtual void fpropOne(Mat &acFeatMaps, const Mat &inFeatMaps)
		{
			cv::max(inFeatMaps, 0, acFeatMaps);
		}

		virtual void bpropOne(Mat &prevLayerDelta, const Mat &acFeatMaps,
							  const Mat &currLayerDelta)
		{
			cv::threshold(acFeatMaps, prevLayerDelta, 0, 1, CV_THRESH_BINARY);
			prevLayerDelta = prevLayerDelta.mul(currLayerDelta);
		}
	};


	// ----------------------------------------------------------------------------
	//
	//					      bounded relu activation function
	//
	// ----------------------------------------------------------------------------
	class BoundReLUFunction : public ActivFunction
	{
	public:
		BoundReLUFunction(const float bound)
		{
			this->bound = bound;
		}

		virtual ~BoundReLUFunction() {}

		virtual void fpropOne(Mat &acFeatMaps, const Mat &inFeatMaps)
		{
			cv::max(inFeatMaps, 0, acFeatMaps);
			cv::min(acFeatMaps, bound, acFeatMaps);
		}

		virtual void bpropOne(Mat &prevLayerDelta, const Mat &acFeatMaps,
							  const Mat &currLayerDelta)
		{
			float maxValue = std::numeric_limits<float>::infinity();
			cv::threshold(prevLayerDelta, acFeatMaps, 0, maxValue, CV_THRESH_TOZERO);
			cv::threshold(acFeatMaps, prevLayerDelta, bound, 1, CV_THRESH_BINARY_INV);
			prevLayerDelta = prevLayerDelta.mul(currLayerDelta);
		}
	private:
		float bound;
	};
}


#endif // activation function
