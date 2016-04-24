#ifndef _CONVNET_UTILITY_PARAM_H_
#define _CONVNET_UTILITY_PARAM_H_
#pragma once

#include <string>

namespace convnet
{
	// --------------------------------------------------------------
	//
	// @brief weight geometry class including basic params for 
	//		  weight
	// 
	//	@param numGroups number of divided groups for weights
	//	@param numWeights number of weights in current layer
	//	@param weightChns number of channels of each weight
	//	@param width window width of each weight
	//	@param height window height of each weight
	//
	// --------------------------------------------------------------
	class WeightGeometry
	{
	public:
		WeightGeometry() : numGroups(1), numWeights(1), weightChns(1), width(1), height(1)
		{}

		WeightGeometry(const int numGroups, const int numWeights,
					   const int weightChns, const int width,
					   const int height, const float initWeightScale)
		{
			this->set(numGroups, numWeights, weightChns, width, height, initWeightScale);
		}

		~WeightGeometry() {}

		inline void set(const int numGroups, const int numWeights,
			            const int weightChns, const int width,
						const int height, const float initWeightScale)
		{
			this->numGroups = numGroups;
			this->numWeights = numWeights;
			this->weightChns = weightChns;
			this->width = width;
			this->height = height;
			this->initWeightScale = initWeightScale;
		}
		
	public:
		int numGroups;
		int numWeights;
		int weightChns;
		int width;
		int height;
		float initWeightScale;
	};


	// --------------------------------------------------------------
	//
	// @brief stride geometry class including basic params for 
	//		  stride
	// 
	//	@param stepRow convoluted step along row direction
	//	@param stepCol convoluted step along col direction
	//
	// --------------------------------------------------------------
	class StrideGeometry
	{
	public:
		StrideGeometry() : stepRow(1), stepCol(1) {}

		StrideGeometry(const int stepRow, const int stepCol)
		{
			this->set(stepRow, stepCol);
		}
		
		~StrideGeometry() {}

		inline void set(const int stepRow, const int stepCol)
		{
			this->stepRow = stepRow;
			this->stepCol = stepCol;
		}

	public:
		int stepRow;
		int stepCol;
	};


	// --------------------------------------------------------------
	//
	// @brief padding geometry class including basic params for 
	//		  padding
	// 
	//	@param top padding top row
	//	@param left padding left col
	//	@param bottom padding bottom row
	//  @param right padding right col
	//
	// --------------------------------------------------------------
	class PadGeometry
	{
	public:
		PadGeometry() : top(0), left(0), bottom(0), right(0) {}

		PadGeometry(const int top, const int left,
					const int bottom, const int right)
		{
			this->set(top, left, bottom, right);
		}

		~PadGeometry() {}

		inline void set(const int top, const int left,
						const int bottom, const int right)
		{
			this->top = top;
			this->left = left;
			this->bottom = bottom;
			this->right = right;
		}

	public:
		int top;
		int left;
		int bottom;
		int right;
	};

	
	// --------------------------------------------------------------
	//
	// @brief learning params class including basic params 
	//		  for learning
	// 
	//
	// --------------------------------------------------------------
	class LearnGeometry
	{
	public:
		LearnGeometry()
			: biasLearningRate(0.01)
			, biasMomentRate(0.95)
			, biasLearningRateScale(0.1)
			, weightLearningRate(0.01)
			, weightMomentRate(0.95)
			, weightLearningRateScale(0.01)
			, weightDecay(0.005)
		{}

		LearnGeometry(const float biasLearningRate, 
					  const float biasMomentRate, 
					  const float biasLearningRateScale, 
					  const float weightLearningRate,
					  const float weightMomentRate,
					  const float weightLearningRateScale,
					  const float weightDecay)

		{
			this->set(biasLearningRate, biasMomentRate,
					  biasLearningRateScale, weightLearningRate,
					  weightMomentRate, weightLearningRateScale,
					  weightDecay);
		}

		~LearnGeometry() {}

		inline void set(const float biasLearningRate,
						const float biasMomentRate,
						const float biasLearningRateScale,
						const float weightLearningRate,
						const float weightMomentRate,
						const float weightLearningRateScale,
						const float weightDecay)
		{
			this->biasLearningRate = biasLearningRate;
			this->biasMomentRate = biasMomentRate;
			this->biasLearningRateScale = biasLearningRateScale;
			this->weightLearningRate = weightLearningRate;
			this->weightMomentRate = weightMomentRate;
			this->weightLearningRateScale = weightLearningRateScale;
			this->weightDecay = weightDecay;
		}

	public:
		float biasLearningRate; // bias learning rate
		float biasMomentRate; // bias moment rate
		float biasLearningRateScale; // bias learning rate scale
		float weightLearningRate; // weight learning rate
		float weightMomentRate; // weight moment rate
		float weightLearningRateScale; // weight learning rate scale
		float weightDecay; // weight decay
	};
}

#endif // Param.h