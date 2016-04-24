#ifndef _CONVNET_CNN_DROPOUTLAYER_H_
#define _CONVNET_CNN_DROPOUTLAYER_H_
#pragma once

#include "../Utility/types.h"
#include "layer.h"
#include <opencv2/core/core.hpp>

namespace convnet
{
	class DropoutLayer : public Layer
	{
	public:
		DropoutLayer() {}

		DropoutLayer(Mat4D &inFeatMaps, const int numThreads = 1);

		virtual ~DropoutLayer();

		inline void setNFCInFeatMaps(Mat4D &inFeatMaps);

		inline void setDropoutRate(const float dropoutRate);

		inline void setMask(const Mat4D &mask, const bool isStaticMask = true);
		
		inline void setNumThreads(const int numThreads = 1);

		inline Mat4D &getNFCOuFeatMaps();

		void init();

		void fprop();

		void bprop();

	protected:
		int numThreads;
		bool isStaticMask;
		float dropoutRate;

	private:
		void fpropOne(Mat3D &ouFeatMaps, const Mat3D &mask, 
					  const Mat3D &inFeatMaps, const bool isStaticMask = false);

		void bpropOne(Mat3D &inFeatMaps, const Mat3D &ouFeatMaps, const Mat3D &mask);


		Mat4D inFeatMaps;
		Mat4D ouFeatMaps;
		Mat4D mask;
	};

	inline void DropoutLayer::setNFCInFeatMaps(Mat4D &inFeatMaps)
	{
		this->inFeatMaps = inFeatMaps;
	}

	inline void DropoutLayer::setDropoutRate(const float dropoutRate)
	{
		this->dropoutRate = dropoutRate;
	}

	inline void DropoutLayer::setMask(const Mat4D &mask, const bool isStaticMask)
	{
		this->mask = mask;
		this->isStaticMask = isStaticMask;
	}

	inline void DropoutLayer::setNumThreads(const int numThreads)
	{
		this->numThreads = numThreads;
	}

	inline Mat4D &DropoutLayer::getNFCOuFeatMaps()
	{
		return this->ouFeatMaps;
	}
}


namespace convnet
{
	using namespace cv;

	class FCDropoutLayer : public DropoutLayer
	{
	public:
		FCDropoutLayer() {}

		FCDropoutLayer(Mat &inFeatMaps, const int numThreads = 1);

		virtual ~FCDropoutLayer();

		inline void setFCInFeatMaps(Mat &inFeatMaps);

		inline void setMask(const Mat &mask, const bool isStaticMask = true);

		inline Mat &getFCOuFeatMaps();

		void init();

		void fprop();

		void bprop();

	private:
		void fpropOne(Mat &ouFeatMaps, Mat &mask,
					  const Mat &inFeatMaps, 
					  const float dropoutRate,
					  const bool isStaticMask = false);

		void bpropOne(Mat &inFeatMaps, const Mat &ouFeatMaps, const Mat &mask);
				
		Mat inFeatMaps;
		Mat ouFeatMaps;
		Mat mask;
	};

	inline void FCDropoutLayer::setFCInFeatMaps(Mat &inFeatMaps)
	{
		this->inFeatMaps = inFeatMaps;
	}

	inline void FCDropoutLayer::setMask(const Mat &mask, const bool isStaticMask)
	{
		this->mask = mask;
		this->isStaticMask = isStaticMask;
	}

	inline Mat &FCDropoutLayer::getFCOuFeatMaps()
	{
		return this->ouFeatMaps;
	}
}


#endif