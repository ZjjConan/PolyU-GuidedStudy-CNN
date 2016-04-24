#ifndef _CONVNET_CNN_UPDATER_H_
#define _CONVNET_CNN_UPDATER_H_
#pragma once

#include "../Utility/types.h"
#include "../Utility/param.h"
#include <opencv2/core/core.hpp>
#include <iostream>

namespace convnet
{	
	using namespace cv;
	using namespace std;

	class Updater
	{
	public:
		Updater() {}
		~Updater() {}

		inline void setUpdaterParams(const float biasLearningRate,
									 const float biasMomentRate,
									 const float biasLearningRateScale,
									 const float weightLearningRate,
									 const float weightMomentRate,
									 const float weightLearningRateScale,
									 const float weightDecay);

		inline float getBiasLearningRate();

		inline float getWeightLearningRate();

		inline float getWeightDecay();

		inline void SGDUpdate(Mat3D &weights, Mat3D &weightMoments,
							  Mat &bias, Mat &biasMoments,
							  const Mat3D &weightGrads,
							  const Mat &biasGrads);

		inline void SGDUpdate(Mat &weights, Mat &weightMoments,
							  Mat &bias, Mat &biasMoments,
							  const Mat &weightGrads,
							  const Mat &biasGrads);

		inline void scaleLearningRate();

	private:
		LearnGeometry lparams;
	};


	inline void Updater::setUpdaterParams(const float biasLearningRate, 
										  const float biasMomentRate, 
										  const float biasLearningRateScale, 
										  const float weightLearningRate, 
										  const float weightMomentRate, 
										  const float weightLearningRateScale, 
										  const float weightDecay)
	{
		lparams.set(biasLearningRate, biasMomentRate, 
					biasLearningRateScale, weightLearningRate, 
					weightMomentRate, weightLearningRateScale, 
					weightDecay);
	}

	inline float Updater::getBiasLearningRate()
	{
		return lparams.biasLearningRate;
	}

	inline float Updater::getWeightLearningRate()
	{
		return lparams.weightLearningRate;
	}

	inline float Updater::getWeightDecay()
	{
		return lparams.weightDecay;
	}


	inline void Updater::SGDUpdate(Mat3D &weights, Mat3D &weightMoments, 
								   Mat &bias, Mat &biasMoments, 
								   const Mat3D &weightGrads, 
								   const Mat &biasGrads)
	{
		float wd = 0;
		if (lparams.weightDecay >= 0)
			wd = lparams.weightDecay;

		// update bias
		biasMoments = lparams.biasMomentRate * biasMoments + lparams.biasLearningRate * (biasGrads + wd * bias);
		bias -= biasMoments;

		// update weights
		for (int g = 0; g < weights.size(); ++g) {
			weightMoments[g] = lparams.weightMomentRate * weightMoments[g] + 
				lparams.weightLearningRate * (weightGrads[g] + wd * weights[g]);
			weights[g] -= weightMoments[g];
		}
	}


	inline void Updater::SGDUpdate(Mat &weights, Mat &weightMoments,
								   Mat &bias, Mat &biasMoments,
								   const Mat &weightGrads,
								   const Mat &biasGrads)
	{
		float wd = 0;
		if (lparams.weightDecay >= 0)
			wd = lparams.weightDecay;

		biasMoments = lparams.biasMomentRate * biasMoments + lparams.biasLearningRate * (biasGrads + wd * bias);
		bias -= biasMoments;

		weightMoments = lparams.weightMomentRate * weightMoments +
			            lparams.weightLearningRate * (weightGrads + wd * weights);
		weights -= weightMoments;

	}

	inline void Updater::scaleLearningRate()
	{
		if (lparams.biasLearningRateScale > 0)
			lparams.biasLearningRate *= lparams.biasLearningRateScale;

		if (lparams.weightLearningRateScale > 0)
			lparams.weightLearningRate *= lparams.weightLearningRateScale;
	}
}

#endif