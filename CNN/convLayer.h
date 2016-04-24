#ifndef _CONVNET_CNN_CONVLAYER_H_
#define _CONVNET_CNN_CONVLAYER_H_
#pragma once

#include "../Utility/types.h"
#include "../utility/param.h"
#include "layer.h"
#include "updater.h"
#include <string>				  // string
#include <vector>				  // vector
#include <opencv2/core/core.hpp>  // cv::Mat

namespace convnet 
{
	using namespace std;
	using namespace cv;

	class ConvLayer : public Layer
	{
	public:
		ConvLayer() {}

		ConvLayer(Mat4D &inFeatMaps, const int numThreads = 1);
		
		~ConvLayer();

		inline void setNFCInFeatMaps(Mat4D &inFeatMaps);

		inline void setWeightGeometry(const int numGroups,
									  const int numWeights,
									  const int weightChns,
									  const int width,
									  const int height,
									  const float initWeightScale);
		
		inline void setStrideGeometry(const int stepRow, const int stepCol);
		
		inline void setPadGeometry(const int top, const int left,
								   const int bottom, const int right);

		inline void setUpdaterParams(const float biasLearningRate,
								     const float biasMomentRate,
									 const float biasLearningRateScale,
									 const float weightLearningRate,
									 const float weightMomentRate,
									 const float weightLearningRateScale,
								     const float weightDecay);
				
		inline void setDzDxFlag(const bool flag);

		inline void setNumThreads(const int numThreads = 1);

		inline Mat4D &getNFCOuFeatMaps();
		
		inline float getCurrObjCost();

		inline Mat3D &getNFCWeights();

		inline Mat &getBias();

		void init();

		void fprop();

		void bprop();

		void update();

		void scaleLearningRate();
		
	private:
		void fpropOne(Mat3D &ouFeatMaps, 
			          const Mat3D &inFeatMaps, 
					  const Mat3D &weights, 
					  const Mat &bias, 
					  const WeightGeometry &wparams,
					  const StrideGeometry &strides,
					  const PadGeometry &padding);

		void bpropOne(Mat3D &prevLayerDelta, 
			          Mat3D &weightGrads, 
					  Mat &biasGrads,
					  const Mat3D &inFeatMaps,
					  const Mat3D &currLayerDelta,
					  const Mat3D &weights, 
					  const WeightGeometry &wparams,
					  const StrideGeometry &strides,
					  const PadGeometry &padding,
					  const bool isDzDx);


	private:
		Mat3D weights;
		Mat3D weightMoments;
		Mat bias;
		Mat biasMoments;
		Mat4D weightGrads;
		Mat3D biasGrads;
		Mat4D inFeatMaps;
		Mat4D ouFeatMaps;
		WeightGeometry wparams;
		StrideGeometry strides;
		PadGeometry padding;
		Updater layerUpdater;

		int numThreads;
		bool isDzDx;
	};


	inline void ConvLayer::setNFCInFeatMaps(Mat4D &inFeatMaps)
	{
		this->inFeatMaps = inFeatMaps;
	}


	inline void ConvLayer::setWeightGeometry(const int numGroups,
											 const int numWeights,
											 const int weightChns,
											 const int width,
											 const int height,
											 const float initWeightScale)
	{
		wparams.set(numGroups, numWeights, weightChns,
					width, height, initWeightScale);
	}

	inline void ConvLayer::setStrideGeometry(const int stepRow, const int stepCol)
	{
		strides.set(stepRow, stepCol);
	}

	inline void ConvLayer::setPadGeometry(const int top, const int left, 
									      const int bottom, const int right)
	{
		padding.set(top, left, bottom, right);
	}

	inline void ConvLayer::setUpdaterParams(const float biasLearningRate, 
											const float biasMomentRate, 
											const float biasLearningRateScale, 
											const float weightLearningRate, 
											const float weightMomentRate, 
											const float weightLearningRateScale, 
											const float weightDecay)
	{
		layerUpdater.setUpdaterParams(biasLearningRate, biasMomentRate,
									  biasLearningRateScale, weightLearningRate, 
									  weightMomentRate, weightLearningRateScale, 
									  weightDecay);
	}
	
	inline void ConvLayer::setDzDxFlag(const bool flag)
	{
		this->isDzDx = flag;
	}

	inline void ConvLayer::setNumThreads(const int numThreads)
	{
		this->numThreads = numThreads;
	}

	inline Mat4D &ConvLayer::getNFCOuFeatMaps()
	{
		return this->ouFeatMaps;
	}

	inline float ConvLayer::getCurrObjCost()
	{
		Mat wwsqsum = Mat::zeros(weights[0].size(), CV_32FC1);
		Mat ww(weights[0].size(), CV_32FC1);
		for (int g = 0; g < weights.size(); ++g) {
			cv::pow(weights[g], 2, ww);
			wwsqsum += ww;
 		}
		return 0.5f * (layerUpdater.getWeightDecay() * (float)sum(wwsqsum)[0]);
	}

	inline Mat3D &ConvLayer::getNFCWeights()
	{
		return this->weights;
	}

	inline Mat &ConvLayer::getBias()
	{
		return this->bias;
	}
}

#endif // convolution layer