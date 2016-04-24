#ifndef _CONVNET_CNN_FCNLAYER_H_
#define _CONVNET_CNN_FCNLAYER_H_
#pragma once

#include "../Utility/param.h"
#include "layer.h"
#include "updater.h"
#include <string>				  // string
#include <vector>				  // vector
#include <opencv2/core/core.hpp>  // cv::Mat

namespace convnet
{
	using namespace std;
	using namespace cv;

	class FCLayer : public Layer
	{
	public:
		FCLayer() {}

		FCLayer(Mat &inFeatMaps, const int numThreads = 1);

		~FCLayer();

		inline void setFCInFeatMaps(Mat &inFeatMaps);

		inline void setWeightGeometry(const int numWeights,
									  const int weightChns,
									  const float initWeightScale);


		inline void setUpdaterParams(const float biasLearningRate,
									 const float biasMomentRate,
									 const float biasLearningRateScale,
									 const float weightLearningRate,
									 const float weightMomentRate,
									 const float weightLearningRateScale,
									 const float weightDecay);

		inline void setDzDxFlag(const bool flag);

		inline void setNumThreads(const int numThreads = 1);

		inline Mat &getFCOuFeatMaps();

		inline float getCurrObjCost();

		inline Mat &getFCWeights();

		inline Mat &getBias();

		void init();

		void fprop();

		void bprop();

		void update();

		void scaleLearningRate();


	private:
		void fpropOne(Mat &ouFeatMaps, const Mat &inFeatMaps,
					  const Mat &weights, const Mat &bias);

		void bpropOne(Mat &prevLayerDelta, 
			          Mat &weightGrads, 
					  Mat &biasGrads,
					  const Mat &inFeatMaps, 
					  const Mat &currLayerDelta, 
					  const Mat &weights, 
					  const bool isDzDx);


	private:
		Mat weights;
		Mat weightMoments;
		Mat bias;
		Mat biasMoments;
		Mat weightGrads;
		Mat biasGrads;
		Mat inFeatMaps;
		Mat ouFeatMaps;
		WeightGeometry wparams;
		Updater layerUpdater;
		
		int numThreads;
		bool isDzDx;
	};


	inline void FCLayer::setFCInFeatMaps(Mat &inFeatMaps)
	{
		this->inFeatMaps = inFeatMaps;
	}

	inline void FCLayer::setWeightGeometry(const int numWeights,
											const int weightChns,
											const float initWeightScale)
	{
		wparams.set(1, numWeights, weightChns, 1, 1, initWeightScale);
	}


	inline void FCLayer::setUpdaterParams(const float biasLearningRate,
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

	inline void FCLayer::setDzDxFlag(const bool flag)
	{
		this->isDzDx = flag;
	}
	
	inline void FCLayer::setNumThreads(const int numThreads)
	{
		this->numThreads = numThreads;
	}

	inline Mat &FCLayer::getFCOuFeatMaps()
	{
		return this->ouFeatMaps;
	}

	inline float FCLayer::getCurrObjCost()
	{
		Mat ww;
		pow(weights, 2, ww);
		return (float)sum(ww)[0] * 0.5f * layerUpdater.getWeightDecay();
	}

	inline Mat &FCLayer::getFCWeights()
	{
		return this->weights;
	}

	inline Mat &FCLayer::getBias()
	{
		return this->bias;
	}
}

#endif // fully-connect layer