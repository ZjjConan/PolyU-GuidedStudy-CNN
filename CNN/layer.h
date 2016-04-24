#ifndef _CVCONVNETS_CNN_LAYER_H_
#define _CVCONVNETS_CNN_LAYER_H_

#include "../Utility/types.h"
#include <opencv2/core/core.hpp>

namespace convnet 
{
	class Layer
	{
	public:
		Layer() {}

		virtual ~Layer() {}

		virtual void setNFCInFeatMaps(Mat4D &inFeatMaps) {}
		
		virtual void setFCInFeatMaps(cv::Mat &inFeatMaps) {}

		virtual void setLabels(cv::Mat &labels) {}

		virtual Mat4D &getNFCOuFeatMaps() { return Mat4D(); }

		virtual cv::Mat &getFCOuFeatMaps() { return cv::Mat(); }

		virtual float getCurrObjCost() { return 0; }

		virtual Mat3D &getNFCWeights() { return Mat3D(); }

		virtual cv::Mat &getFCWeights() { return cv::Mat(); }

		virtual cv::Mat &getBias() { return cv::Mat(); }

		virtual void init() = 0;

		virtual void fprop() = 0;
		
		virtual void bprop() = 0;

		virtual void update() {}

		virtual void scaleLearningRate() {}
	};
}

#endif