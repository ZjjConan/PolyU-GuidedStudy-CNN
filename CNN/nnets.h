/**
 *
 */
#ifndef _CONVNET_CNN_NNETS_H_
#define _CONVNET_CNN_NNETS_H_

#include "../Utility/types.h"
#include "../Utility/check.h"
#include "../Utility/param.h"
#include "layer.h"
#include "activLayer.h"
#include "concatLayer.h"
#include "convLayer.h"
#include "fcLayer.h"
#include "poolLayer.h"
#include "dropoutLayer.h"
#include "updater.h"
#include "loss.h"
#include <string>
#include <opencv2/core/core.hpp>

namespace convnet 
{	
	using namespace std;
	using namespace cv;

	class NNets
	{
	public:
		NNets();
	
		~NNets();
		
		inline Layer *getLayerNode(const int index);

		inline string getLayerName(const int index);

		inline void setInputImages(Mat4D &inFeatMaps);

		inline void setInputLabels(Mat &labels);
		
		inline void addLayer(Layer *func, const string name);

		inline int getNumberLayers();

		inline long int getNumberParams();
		
		inline float getModelSize();

		inline float getCurrObjCost();

		// create a convolution layer
		void createConvLayer(const WeightGeometry &wparams, const StrideGeometry &strides,
						     const PadGeometry &padding, const LearnGeometry &lparams,
							 const bool isDzDx, const int numThreads = 1);
			

		// create a pooling layer
		void createPoolLayer(const WeightGeometry &wparams, const StrideGeometry &strides,
						     const PadGeometry &padding, const string poolMethod, 
							 const bool isScaledMaps = false, const int numThreads = 1);


		// create a activation layer
		void createActivLayer(const string activFuncName, const int numThreads = 1);

		// create a FC activation layer
		void createFCActivLayer(const string activFuncName, const int numThreads = 1);
		
		// create a concatenation layer
		void createConcatLayer(const WeightGeometry &wparams, const int numThreads = 1);

		// create a fully-connected layer
		void createFCNLayer(const WeightGeometry &wparams, const LearnGeometry &lparams,
							const bool isDzDx, const int numThreads = 1);

		
		// create a dropout layer
		void createDropoutLayer(const bool isStatisMask, const float dropoutRate,
							    const int numThreads = 1, const Mat4D &mask = Mat4D ());

		// create a FC dropout layer
		void createFCDropoutLayer(const bool isStatisMask, const float dropoutRate,
							      const int numThreads = 1, const Mat &mask = Mat());

		// create a loss layer
		void createLossLayer(const int numThreads = 1);

		// init each layer and build nodes chains 
		void builChains(const bool isRebuild = false);
				
		// forward pass
		void fprop();
		
		// backward pass
		void bprop();

		// update learning params
		void update();

		// scale learning rate
		void scaleLearningRate();

		// remove dropout layer and rebuild node chains
		void removeDropoutLayer();

		// release model
		void release();

	private:
		vector<Layer *> nodeFunc;
		vector<string > nodeName;
	};


	inline Layer *NNets::getLayerNode(const int index)
	{
		return nodeFunc[index];
	}

	inline string NNets::getLayerName(const int index)
	{
		return nodeName[index];
	}

	inline void NNets::setInputImages(Mat4D &inFeatMaps)
	{
		nodeFunc[0]->setNFCInFeatMaps(inFeatMaps);
	}

	inline void NNets::setInputLabels(Mat &labels)
	{
		int numNodes = nodeFunc.size();
		nodeFunc[numNodes - 1]->setLabels(labels);
	}

	inline void NNets::addLayer(Layer *func, const string name)
	{
		nodeFunc.push_back(func);
		nodeName.push_back(name);
	}

	inline int NNets::getNumberLayers()
	{
		return nodeFunc.size();
	}

	inline long int NNets::getNumberParams()
	{
		long int num = 0;
		for (int i = 0; i < nodeName.size(); ++i) {
			if (nodeName[i] == "conv")
			{
				int numGroups = nodeFunc[i]->getNFCWeights().size();
				for (int g = 0; g < numGroups; ++g)
					num += nodeFunc[i]->getNFCWeights()[g].total();
				num += nodeFunc[i]->getBias().total();
			}
			else if (nodeName[i] == "fc") {
				num += nodeFunc[i]->getFCWeights().total();
				num += nodeFunc[i]->getBias().total();
			}
		}
		return num;
	}

	inline float NNets::getModelSize()
	{
		float num = getNumberParams();
		num *= 4;
		num /= 1024;
		num /= 1024;
		return num;
	}

	inline float NNets::getCurrObjCost()
	{
		float objCost = 0.0f;
		for (int i = 0; i < nodeName.size(); ++i) {
			objCost += nodeFunc[i]->getCurrObjCost();
		}
		return objCost;
	}
}


#endif // _convnet_cnn_nnets_h_