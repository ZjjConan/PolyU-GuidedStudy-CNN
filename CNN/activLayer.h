#ifndef _CONVNET_CNN_ACTIVLAYERR_H_
#define _CONVNET_CNN_ACTIVLAYERR_H_
#pragma once

#include "../Utility/types.h"
#include "activFunc.h"
#include "layer.h"
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

namespace convnet
{
	class ActivLayer : public Layer
	{
	public:
		ActivLayer() {}

		ActivLayer(Mat4D &inFeatMaps, const int numThreads = 1);
		
		virtual ~ActivLayer();

		inline void setNFCInFeatMaps(Mat4D &inFeatMaps);

		inline void setActivFuncName(const string &activFuncName);

		inline void setNumThreads(const int numThreads = 1);

		inline ActivFunction *getActivFunction(const string &activFuncName);

		inline Mat4D &getNFCOuFeatMaps();

		void init();

		void fprop();
		
		void bprop();
		
	protected:
		int numThreads;
		string activFuncName;
		ActivFunction *activFunc;

	private:
		void fpropOne(Mat3D &tmFeatMaps, const Mat3D &inFeatMaps, ActivFunction *func);

		void bpropOne(Mat3D &inFeatMaps, const Mat3D &tmFeatMaps, const Mat3D &ouFeatMaps,
			          ActivFunction *func);

		void copyToOutputMaps(Mat3D &ouFeatMaps, const Mat3D &tmFeatMaps);

	private:
		Mat4D inFeatMaps;
		Mat4D tmFeatMaps;
		Mat4D ouFeatMaps;
	};


	inline void ActivLayer::setNFCInFeatMaps(Mat4D &inFeatMaps)
	{
		this->inFeatMaps = inFeatMaps;
	}

	inline void ActivLayer::setActivFuncName(const string &activFuncName)
	{
		this->activFuncName = activFuncName;
	}

	inline void ActivLayer::setNumThreads(const int numThreads)
	{
		this->numThreads;
	}

	inline Mat4D &ActivLayer::getNFCOuFeatMaps()
	{
		return this->ouFeatMaps;
	}


	inline ActivFunction *ActivLayer::getActivFunction(const string &activFuncName)
	{
		if (!_strcmpi("Linear", activFuncName.c_str()))
			return new LinearFunction;

		else if (!_strcmpi("Sigmoid", activFuncName.c_str()))
			return new SigmoidFunction;

		else if (!_strcmpi("ReLU", activFuncName.c_str()))
			return new ReLUFunction;

		//[TODO params.bound in BoundReLU]
		else if (!_strcmpi("BReLU", activFuncName.c_str()))
			return new BoundReLUFunction(1);

		else return NULL;
	}

}



namespace convnet
{
	class FCActivLayer : public ActivLayer
	{
	public:
		FCActivLayer() {}

		FCActivLayer(Mat &inFeatMaps, const int numThreads = 1);

		virtual ~FCActivLayer();

		inline void setFCInFeatMaps(Mat &inFeatMaps);

		inline void setNumThreads(const int numThreads = 1);

		inline Mat &getFCOuFeatMaps();

		void init();

		void fprop();

		void bprop();

	private:
		void fpropOne(Mat &tmFeatMaps, const Mat &inFeatMaps, ActivFunction *func);

		void bpropOne(Mat &inFeatMaps, const Mat &tmFeatMaps, const Mat &ouFeatMaps,
			          ActivFunction *func);

		void copyToOutputMaps(Mat &ouFeatMaps, const Mat &tmFeatMaps);

	private:
		Mat inFeatMaps;
		Mat tmFeatMaps;
		Mat ouFeatMaps;
		int numThreads;
	};

	inline void FCActivLayer::setFCInFeatMaps(Mat &inFeatMaps)
	{
		this->inFeatMaps = inFeatMaps;
	}

	inline void FCActivLayer::setNumThreads(const int numThreads)
	{
		this->numThreads = numThreads;
	}

	inline Mat &FCActivLayer::getFCOuFeatMaps()
	{
		return this->ouFeatMaps;
	}
}

#endif // activation layer