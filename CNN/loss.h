#ifndef _CONVNET_CNN_LOSS_H_
#define _CONVNET_CNN_LOSS_H_

#include "../Utility/param.h"
#include "layer.h"
#include <string>
#include <vector>
#include <opencv2/core/core.hpp> 


namespace convnet
{
	using namespace std;
	using namespace cv;
	
	class SoftmaxLoss : public Layer
	{
	public:
		SoftmaxLoss() {}

		SoftmaxLoss(Mat &inFeatMaps, const Mat &labels, const int numThreads = 1);

		virtual ~SoftmaxLoss();

		inline void setFCInFeatMaps(Mat &inFeatMaps);

		inline void setLabels(Mat &labels);

		inline void setNumThreads(int numThreads = 1);

		inline float getCurrObjCost();
		
		inline Mat &getFCOuFeatMaps();

		void init();

		void fprop();

		void bprop();

	private:
		void label2matrix(Mat &lmat, const Mat &label);

		void fpropOne(Mat &inFeatMaps);

		void bpropOne(Mat &delta, const Mat &inFeatMaps, const Mat &lmat);

	private:
		Mat inFeatMaps;
		Mat labels;
		Mat lmat;
		int numThreads;
	};
	

	inline void SoftmaxLoss::setFCInFeatMaps(Mat &inFeatMaps)
	{
		this->inFeatMaps = inFeatMaps;
	}

	inline void SoftmaxLoss::setLabels(Mat &labels)
	{
		this->labels = labels;
	}

	inline void SoftmaxLoss::setNumThreads(int numThreads)
	{
		this->numThreads = numThreads;
	}

	inline Mat &SoftmaxLoss::getFCOuFeatMaps()
	{
		return this->inFeatMaps;
	}

	inline float SoftmaxLoss::getCurrObjCost()
	{
		Mat logScores;
		log(inFeatMaps, logScores);
		logScores = logScores.mul(lmat);
		return (float)(-(sum(logScores)[0]) / lmat.rows);
	}
}

#endif // _convnet_include_classifier_h_