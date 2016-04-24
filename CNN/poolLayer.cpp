/*
*/

#include "../Utility/check.h"
#include "poolLayer.h"

#include <algorithm>
#include <climits>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace convnet
{
	using namespace std;
	using namespace cv;
	

	class MaxOperator : public OperatorFunction
	{
	public:
		MaxOperator() {}
		virtual ~MaxOperator() {}

		virtual void fprop(float *ouFeatMaps,
						   const float *inFeatMaps, 
						   const int inCols,
						   const int r1, 
						   const int r2, 
						   const int c1, 
						   const int c2);

		virtual void bprop(float *upFeatMaps, 
						   const float *inFeatMaps,
						   const float *ouFeatMaps,
						   const int inCols,
						   const int r1, 
						   const int r2, 
						   const int c1, 
						   const int c2);
	};

	void MaxOperator::fprop(float *ouFeatMaps,
							const float *inFeatMaps,
						    const int inCols,
						    const int r1,
						    const int r2,
							const int c1,
							const int c2)
	{
		float maxValue = -std::numeric_limits<float>::infinity();
		for (int r = r1; r < r2; ++r) {
 			for (int c = c1; c < c2; ++c) {
				if (maxValue < CV_MAT_AT(inFeatMaps, r, c, inCols))
					maxValue = CV_MAT_AT(inFeatMaps, r, c, inCols);
 			}
		}
		*ouFeatMaps = maxValue;
	}

	void MaxOperator::bprop(float *upFeatMaps,
						    const float *inFeatMaps,
							const float *ouFeatMaps,
						    const int inCols,
							const int r1,
						    const int r2,
						    const int c1,
							const int c2)
	{
		int maxR = 0;
		int maxC = 0;
		float maxValue = -std::numeric_limits<float>::infinity();
		for (int r = r1; r < r2; ++r) {
			for (int c = c1; c < c2; ++c) {
				if (maxValue < CV_MAT_AT(inFeatMaps, r, c, inCols)) {
					maxValue = CV_MAT_AT(inFeatMaps, r, c, inCols);
					maxR = r;
					maxC = c;
				}
			}
		}
		CV_MAT_AT(upFeatMaps, maxR, maxC, inCols) += *ouFeatMaps;
	}




	class AvgOperator : public OperatorFunction
	{
	public:
		AvgOperator() {}
		virtual ~AvgOperator() {}

		virtual void fprop(float *ouFeatMaps,
						   const float *inFeatMaps,
						   const int inCols,
						   const int r1,
						   const int r2,
						   const int c1,
						   const int c2);

		virtual void bprop(float *upFeatMaps,
						   const float *inFeatMaps,
						   const float *ouFeatMaps,
						   const int inCols,
						   const int r1,
						   const int r2,
						   const int c1,
						   const int c2);
	};

	void AvgOperator::fprop(float *ouFeatMaps,
						    const float *inFeatMaps,
							const int inCols,
							const int r1,
						    const int r2,
						    const int c1,
							const int c2)
	{
		float accValue = 0.0f;
		for (int r = r1; r < r2; ++r) {
			for (int c = c1; c < c2; ++c) {
				accValue += CV_MAT_AT(inFeatMaps, r, c, inCols);
			}
		}
		*ouFeatMaps = accValue;
	}

	void AvgOperator::bprop(float *upFeatMaps,
							const float *inFeatMaps,
						    const float *ouFeatMaps,
							const int inCols,
							const int r1,
							const int r2,
							const int c1,
							const int c2)
	{
		for (int r = r1; r < r2; ++r) {
			for (int c = c1; c < c2; ++c) {
				CV_MAT_AT(upFeatMaps, r, c, inCols) += *ouFeatMaps;
			}
		}
	}
}



namespace convnet
{
	using namespace std;
	using namespace cv;

	PoolLayer::PoolLayer(Mat4D &inFeatMaps, const int numThreads) 
	{
		this->inFeatMaps = inFeatMaps;
		this->numThreads = numThreads;
	}

	PoolLayer::~PoolLayer()
	{
		inFeatMaps.clear();
		ouFeatMaps.clear();
		if (poolOpt != NULL) {
			free(poolOpt);
			poolOpt = NULL;
		}
	}


	void PoolLayer::init()
	{
		// set pooling operator
		if (!_strcmpi(poolMethod.c_str(), "Max"))
			poolOpt = new MaxOperator;
		else if (!_strcmpi(poolMethod.c_str(), "Avg"))
			poolOpt = new AvgOperator;
		else
			argu::ASSERT(true, " pooling operation wrong: only support Max / Avg / Sub !\n");


		// allocate space
		int numImages = inFeatMaps.size();
		int chns = inFeatMaps[0].size();
		int rows = inFeatMaps[0][0].rows;
		int cols = inFeatMaps[0][0].cols;
		int ouRows = (rows + padding.bottom + padding.top - wparams.height) / strides.stepRow + 1;
		int ouCols = (cols + padding.left + padding.right - wparams.width) / strides.stepCol + 1;
		
		ouFeatMaps.resize(numImages);
		for (int i = 0; i < numImages; ++i) {
			ouFeatMaps[i].resize(chns);
			for (int ch = 0; ch < chns; ++ch) {
				ouFeatMaps[i][ch] = Mat::zeros(ouRows, ouCols, CV_32FC1);
			}
		}
	}

	void PoolLayer::fprop()
	{
		int numImages = inFeatMaps.size();

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		
		for (int i = 0; i < numImages; ++i) {
			fpropOne(ouFeatMaps[i], inFeatMaps[i], 
					 wparams, strides, padding, poolOpt);
		}

		if (isScaledMaps) {

			float pscale = wparams.height * wparams.width;

			#ifdef _OPENMP
			#pragma omp parallel for
			#endif

			for (int i = 0; i < numImages; ++i)
				scaleMaps(ouFeatMaps[i], pscale);
		}
	}

	void PoolLayer::bprop()
	{
		int numImages = inFeatMaps.size();
		
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif

		for (int i = 0; i < numImages; ++i) {
			bpropOne(inFeatMaps[i], inFeatMaps[i], ouFeatMaps[i], 
					 wparams, strides, padding, poolOpt);
		}

		if (isScaledMaps) {

			float pscale = wparams.height * wparams.width;
			
			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			
			for (int i = 0; i < numImages; ++i) 
				scaleMaps(inFeatMaps[i], pscale);
		}
	}


	// ----------------------------------------------------------------------------
	//
	//								private function impl
	//
	// ----------------------------------------------------------------------------
	void PoolLayer::fpropOne(Mat3D &ouFeatMaps, 
						     const Mat3D &inFeatMaps,
							 const WeightGeometry &wparams,
							 const StrideGeometry &strides,
							 const PadGeometry &padding,
						     OperatorFunction *func)
	{
		int inChns = inFeatMaps.size();
		int inRows = inFeatMaps[0].rows;
		int inCols = inFeatMaps[0].cols;
		int ouRows = ouFeatMaps[0].rows;
		int ouCols = ouFeatMaps[0].cols;
		float *ouMapPtr = NULL;
		float *inMapPtr = NULL;

		int r1, r2, c1, c2;
		for (int ch = 0; ch < inChns; ++ch) {
			ouMapPtr = CV_MAT_PRF(ouFeatMaps[ch]);
 			inMapPtr = CV_MAT_PRF(inFeatMaps[ch]);
			for (int r = 0; r < ouRows; ++r) {
				r1 = r * strides.stepRow - padding.top;
				r2 = min(r1 + wparams.height, inRows);
				r1 = max(r1, 0);
				r2 = max(r2, 0);
				for (int c = 0; c < ouCols; ++c) {
					c1 = c * strides.stepCol - padding.left;
					c2 = min(c1 + wparams.width, inCols);
					c1 = max(c1, 0);
					c2 = max(c2, 0);
					func->fprop(&ouMapPtr[r * ouCols + c], inMapPtr, inCols, 
								 r1, r2, c1, c2);
				}
			}
		}
		ouMapPtr = NULL;
		inMapPtr = NULL;
	}


	void PoolLayer::bpropOne(Mat3D &upFeatMaps,
						     const Mat3D &inFeatMaps,
						     const Mat3D &ouFeatMaps,
							 const WeightGeometry &wparams,
							 const StrideGeometry &strides,
							 const PadGeometry &padding,
						     OperatorFunction *func)
	{
		int chns = inFeatMaps.size();
		int inRows = inFeatMaps[0].rows;
		int inCols = inFeatMaps[0].cols;
		int ouRows = ouFeatMaps[0].rows;
		int ouCols = ouFeatMaps[0].cols;
		Mat tmp(inRows, inCols, upFeatMaps[0].type());
		float *ouMapPtr = NULL;
		float *inMapPtr = NULL;
		float *tmMapPtr = CV_MAT_PRF(tmp);
		
		int r1, r2, c1, c2;
		for (int ch = 0; ch < chns; ++ch) {
			memset(tmMapPtr, 0, inRows * inCols * sizeof(float));
			ouMapPtr = CV_MAT_PRF(ouFeatMaps[ch]);
			inMapPtr = CV_MAT_PRF(inFeatMaps[ch]);
			for (int r = 0; r < ouRows; ++r) {
				r1 = r * strides.stepRow - padding.top;
				r2 = min(r1 + wparams.height, inRows);
				r1 = max(r1, 0);
				r2 = max(r2, 0);
				for (int c = 0; c < ouCols; ++c) {
					c1 = c * strides.stepCol - padding.left;
					c2 = min(c1 + wparams.width, inCols);
					c1 = max(c1, 0);
					c2 = max(c2, 0);
					func->bprop(tmMapPtr, inMapPtr, &ouMapPtr[r * ouCols + c], inCols, 
								r1, r2, c1, c2);
				}
			}
			memcpy(upFeatMaps[ch].data, tmp.data, inRows * inCols * sizeof(float));
		}
		ouMapPtr = NULL;
		inMapPtr = NULL;
	}

	void PoolLayer::scaleMaps(vector<Mat> &inoufeatMaps, const float area)
	{
		for (int ch = 0; ch < inoufeatMaps.size(); ch++)
			inoufeatMaps[ch] /= area;
	}
}