/*
 */

#include "nnets.h"

namespace convnet
{
	NNets::NNets() {}

	NNets::~NNets()
	{
		nodeFunc.clear();
		nodeName.clear();
	}

	void NNets::createConvLayer(const WeightGeometry &wparams, const StrideGeometry &strides,
								const PadGeometry &padding, const LearnGeometry &lparams,
								const bool isDzDx, const int numThreads)
	{
		ConvLayer *currNode = new ConvLayer;
		currNode->setWeightGeometry(wparams.numGroups, wparams.numWeights, wparams.weightChns, 
								    wparams.width, wparams.height, wparams.initWeightScale);
		currNode->setStrideGeometry(strides.stepRow, strides.stepCol);
		currNode->setPadGeometry(padding.top, padding.left, padding.bottom, padding.right);
		currNode->setUpdaterParams(lparams.biasLearningRate, lparams.biasMomentRate,
								   lparams.biasLearningRateScale, lparams.weightLearningRate,
								   lparams.weightMomentRate, lparams.weightLearningRateScale,
								   lparams.weightDecay);
		currNode->setDzDxFlag(isDzDx);
		currNode->setNumThreads(numThreads);

		// add-in nnets nodes
		addLayer(currNode, "conv");
	}


	void NNets::createPoolLayer(const WeightGeometry &wparams, const StrideGeometry &strides,
							    const PadGeometry &padding, const string poolMethod,
								const bool isScaledMaps, const int numThreads)
	{
		PoolLayer *currNode = new PoolLayer;
		currNode->setWeightGeometry(wparams.width, wparams.height);
		currNode->setStrideGeometry(strides.stepRow, strides.stepCol);
		currNode->setPadGeometry(padding.top, padding.left, padding.bottom, padding.right);
		currNode->setPoolMethod(poolMethod);
		currNode->setScaleMapFlag(isScaledMaps);
		currNode->setNumThreads();
		
		// add-in nnets nodes
		addLayer(currNode, "pool");
	}

	void NNets::createActivLayer(const string activFuncName, const int numThreads)
	{
		ActivLayer *currNode = new ActivLayer;
		currNode->setActivFuncName(activFuncName);
		currNode->setNumThreads(numThreads);

		// add-in nnets nodes
		addLayer(currNode, "activ");
	}

	void NNets::createFCActivLayer(const string activFuncName, const int numThreads)
	{
		FCActivLayer *currNode = new FCActivLayer;
		currNode->setActivFuncName(activFuncName);
		currNode->setNumThreads(numThreads);

		// add-in nnets nodes
		addLayer(currNode, "fcActiv");
	}

	void NNets::createConcatLayer(const WeightGeometry &wparams, const int numThreads)
	{
		ConcatLayer *currNode = new ConcatLayer;
		currNode->setWeightGeometry(wparams.width, wparams.height);
		currNode->setNumThreads(numThreads);

		// add-in nnets nodes
		addLayer(currNode, "concat");
	}


	void NNets::createFCNLayer(const WeightGeometry &wparams, const LearnGeometry &lparams,
							   const bool isDzDx, const int numThreads)
	{
		FCLayer *currNode = new FCLayer;
		currNode->setWeightGeometry(wparams.numWeights, wparams.weightChns, wparams.initWeightScale);
		currNode->setUpdaterParams(lparams.biasLearningRate, lparams.biasMomentRate,
								   lparams.biasLearningRateScale, lparams.weightLearningRate,
								   lparams.weightMomentRate, lparams.weightLearningRateScale,
								   lparams.weightDecay);
		currNode->setDzDxFlag(isDzDx);
		currNode->setNumThreads(numThreads);

		// add-in nnets nodes
		addLayer(currNode, "fc");
	}

	void NNets::createDropoutLayer(const bool isStatisMask, const float dropoutRate,
								   const int numThreads, const Mat4D &mask)
	{
		DropoutLayer *currNode = new DropoutLayer;
		currNode->setMask(mask, isStatisMask);
		currNode->setDropoutRate(dropoutRate);
		currNode->setNumThreads(numThreads);

		// add-in nnets nodes
		addLayer(currNode, "dropout");
	}

	void NNets::createFCDropoutLayer(const bool isStatisMask, const float dropoutRate, 
									 const int numThreads , const Mat &mask)
	{
		FCDropoutLayer *currNode = new FCDropoutLayer;
		currNode->setMask(mask, isStatisMask);
		currNode->setDropoutRate(dropoutRate);
		currNode->setNumThreads(numThreads);

		// add-in nnets nodes
		addLayer(currNode, "fcDropout");
	}


	void NNets::createLossLayer(const int numThreads)
	{
		SoftmaxLoss *currNode = new SoftmaxLoss;
		currNode->setNumThreads(numThreads);

		// add-in nnets nodes
		addLayer(currNode, "loss");
	}

	void NNets::builChains(const bool isRebuild)
	{
		NNETS_INIT(nodeFunc, nodeName);

		if (isRebuild)
			nodeFunc[0]->init();

		for (int i = 1; i < nodeName.size(); ++i) {
			if (nodeName[i] == "conv" || nodeName[i] == "pool" || nodeName[i] == "activ" ||
				nodeName[i] == "dropout" || nodeName[i] == "concat")
				nodeFunc[i]->setNFCInFeatMaps(nodeFunc[i - 1]->getNFCOuFeatMaps());
				
			else if (nodeName[i] == "fcActiv" || nodeName[i] == "fcDropout" || 
				     nodeName[i] == "fc")
				nodeFunc[i]->setFCInFeatMaps(nodeFunc[i - 1]->getFCOuFeatMaps());
	
			else if (nodeName[i] == "loss") {
				nodeFunc[i]->setFCInFeatMaps(nodeFunc[i - 1]->getFCOuFeatMaps());
			}

			// initialize current node
			if (isRebuild)
				nodeFunc[i]->init();
		}
	}
	


	void NNets::fprop()
	{
		NNETS_INIT(nodeFunc, nodeName);
		for (int i = 0; i < nodeName.size(); ++i) {
			nodeFunc[i]->fprop();
 		}
	}

	void NNets::bprop()
	{
		NNETS_INIT(nodeFunc, nodeName);

		for (int i = nodeName.size() - 1; i >= 0; --i) {
			nodeFunc[i]->bprop();
		}
	}

	void NNets::update()
	{
		NNETS_INIT(nodeFunc, nodeName);

		for (int i = 0; i < nodeName.size(); ++i) {
			nodeFunc[i]->update();
		}
	}

	void NNets::scaleLearningRate()
	{
		NNETS_INIT(nodeFunc, nodeName);

		for (int i = 0; i < nodeName.size(); ++i) {
			if (nodeName[i] == "conv" || nodeName[i] == "fc")
				nodeFunc[i]->scaleLearningRate();
		}
	}

	void NNets::removeDropoutLayer()
	{
		NNETS_INIT(nodeFunc, nodeName);

		
		while (1) {
			// find out dropout layers
			int i = 0;
			for (i; i < nodeName.size(); ++i) {
				if (nodeName[i] == "dropout" || nodeName[i] == "fcDropout")
					break;
			}

			// no dropout layer 
			if (i == nodeName.size())
				break;

			nodeName.erase(nodeName.begin() + i);
			nodeFunc.erase(nodeFunc.begin() + i);

			// rebuild chains
			builChains(false);
		}
	}

	void NNets::release()
	{
		if (!nodeFunc.empty()) {
			for (int i = 0; i < nodeName.size(); ++i) {
				if (nodeFunc[i] != NULL) delete nodeFunc[i];
			}
		}
		nodeFunc.clear();
		nodeName.clear();
	}
}