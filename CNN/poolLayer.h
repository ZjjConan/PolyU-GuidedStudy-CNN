#ifndef _CONVNET_CNN_POOLLAYER_H_
#define _CONVNET_CNN_POOLLAYER_H_
#pragma once

#include "../Utility/types.h"
#include "../utility/param.h"
#include "layer.h"
#include <cassert>   // assert 
#include <string>    // string
#include <vector>    // vector
#include <opencv2/core/core.hpp> // Mat

namespace convnet
{
	using namespace std;
	using namespace cv;

	// TODO
	// pooling function interface
	class OperatorFunction
	{
	public:
		virtual ~OperatorFunction() {}

		virtual void fprop(float *ouFeatMaps, const float *inFeatMaps, const int inCols,
						   const int r1, const int r2, const int c1, const int c2) = 0;

		virtual void bprop(float *upFeatMaps, const float *inFeatMap, const float *ouFeatMaps,
						   const int inCols, const int r1, const int r2, const int c1, const int c2) = 0;
	};



	class PoolLayer : public Layer
	{
	public:
		PoolLayer() {}

		PoolLayer(Mat4D &inFeatMaps, const int numThreads = 1);

		~PoolLayer();

		inline void setNFCInFeatMaps(Mat4D &inFeatMaps);

		inline void setWeightGeometry(const int width, const int height);

		inline void setStrideGeometry(const int stepRow, const int stepCol);

		inline void setPadGeometry(const int top, const int left, const int bottom, const int right);

		inline void setPoolMethod(const string poolMethod);

		inline void setScaleMapFlag(const bool isScaledMaps);

		inline void setNumThreads(const int numThreads = 1);

		inline Mat4D &getNFCOuFeatMaps();

		void init();

		void fprop();

		void bprop();

	private:
		void fpropOne(Mat3D &ouFeatMaps,
					  const Mat3D &inFeatMaps,
					  const WeightGeometry &wparams,
					  const StrideGeometry &strides,
					  const PadGeometry &padding,
					  OperatorFunction *func);

		void bpropOne(Mat3D &upFeatMaps,
					  const Mat3D &inFeatMaps,
					  const Mat3D &ouFeatMaps,
					  const WeightGeometry &wparams,
					  const StrideGeometry &strides,
					  const PadGeometry &padding,
					  OperatorFunction *func);

		void scaleMaps(Mat3D &upFeatMaps, const float area);


	private:
		// feature maps
		Mat4D inFeatMaps;
		Mat4D ouFeatMaps;
		WeightGeometry wparams;
		StrideGeometry strides;
		PadGeometry padding;
		OperatorFunction *poolOpt;

		string poolMethod;
		bool isScaledMaps;
		int numThreads;
	};

	inline void PoolLayer::setNFCInFeatMaps(Mat4D &inFeatMaps)
	{
		this->inFeatMaps = inFeatMaps;
	}

	inline void PoolLayer::setWeightGeometry(const int width, const int height)
	{
		wparams.set(-1, -1, -1, width, height, -1);
	}

	inline void PoolLayer::setStrideGeometry(const int stepRow, const int stepCol)
	{
		strides.set(stepRow, stepCol);
	}

	inline void PoolLayer::setPadGeometry(const int top, const int left,
										  const int bottom, const int right)
	{
		padding.set(top, left, bottom, right);
	}

	inline void PoolLayer::setPoolMethod(const string poolMethod)
	{
		this->poolMethod = poolMethod;
	}

	inline void PoolLayer::setScaleMapFlag(const bool isScaledMaps)
	{
		this->isScaledMaps = isScaledMaps;
	}

	inline void PoolLayer::setNumThreads(const int numThreads)
	{
		this->numThreads = numThreads;
	}

	inline Mat4D &PoolLayer::getNFCOuFeatMaps()
	{
		return this->ouFeatMaps;
	}
}


#endif // pooling layer