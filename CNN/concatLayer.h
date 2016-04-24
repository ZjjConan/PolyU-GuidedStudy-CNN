#ifndef _CONVNET_CNN_CONCATLAYER_H_
#define _CONVNET_CNN_CONCATLAYER_H_
#pragma once

#include "../Utility/types.h"
#include "../utility/param.h"
#include "layer.h"
#include <string>				  // string
#include <vector>				  // vector
#include <opencv2/core/core.hpp>  // cv::Mat

namespace convnet
{
	using namespace std;
	using namespace cv;

	class ConcatLayer : public Layer
	{
	public:
		ConcatLayer() {}
		
		ConcatLayer(Mat4D &inFeatMaps, const int numThreads = 1);

		~ConcatLayer();
		
		inline void setNFCInFeatMaps(Mat4D &inFeatMaps);

		inline void setWeightGeometry(const int width, const int height);

		inline void setNumThreads(const int numThreads = 1);

		inline Mat &getFCOuFeatMaps();

		void init();

		void fprop();

		void bprop();

	private:
		void fpropOne(Mat &ouFeatMaps, const Mat3D &inFeatMaps, const WeightGeometry &wparams);

		void bpropOne(Mat3D &inFeatMaps, const Mat &ouFeatMaps, const WeightGeometry &wparams);


	private:
		Mat4D inFeatMaps;
		Mat ouFeatMaps;
		WeightGeometry wparams;
		int numThreads;
	};


	inline void ConcatLayer::setNFCInFeatMaps(Mat4D &inFeatMaps)
	{
		this->inFeatMaps = inFeatMaps;
	}

	inline void ConcatLayer::setWeightGeometry(const int width, const int height)
	{
		wparams.set(-1, -1, -1, width, height, -1);
	}

	inline void ConcatLayer::setNumThreads(const int numThreads)
	{
		this->numThreads = numThreads;
	}

	inline Mat &ConcatLayer::getFCOuFeatMaps()
	{
		return this->ouFeatMaps;
	}
}

#endif // concatenation layer