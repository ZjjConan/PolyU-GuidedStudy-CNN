#ifndef _CONVNET_UTILITY_CHECK_H_
#define _CONVNET_UTILITY_CHECK_H_
#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <opencv2/core/core.hpp>

namespace argu
{
	inline void ASSERT(bool cond, const char *msg)
	{
		if (cond) {
			fprintf(stderr, "Error:%s\n", msg);
			exit(-1);
		}
	}

	inline void ABORT(bool cond)
	{
		if (cond) {
			fprintf(stderr, "");
			exit(-1);
		}
	}
}

namespace convnet 
{
	
	// @brief CV_MAT_PR get point of cv::Mat
	#ifndef CV_MAT_PR
	#define CV_MAT_PR(mat, DType) (DType *)(mat.data)
	#endif

	#ifndef CV_MAT_PRF
	#define CV_MAT_PRF(mat) CV_MAT_PR(mat, float)
	#endif

	// @brief CV_MAT_AT get value of cv::Mat, should be first get point by CV_MAT_PR
	#ifndef CV_MAT_AT
	#define CV_MAT_AT(matdata, r, c, cols) (matdata)[(r) * (cols) + (c)]
	#endif


	// @brief it must be faster by directly access point of cv::Mat 
	#ifndef CV_MAT_PRAT
	#define CV_MAT_PRAT(mat, r, c, DType) ((DType *)(mat.data))[(r)*(mat.cols) + (c)]
	#endif
	

	// --------------------------------------------------------------------------
	//	
	//							Layer Check Macro
	//
	// --------------------------------------------------------------------------


	// @brief non-fc layer input feature maps check
	#ifndef NONFC_INPUT_INIT
	#define NONFC_INPUT_INIT(inFeatMaps) \
	{ \
		argu::ASSERT(inFeatMaps.empty() == true, \
					 " input feature maps should not be empty !\n"); \
		argu::ASSERT(inFeatMaps[0].empty() == true, \
					 " input feature maps do not have legal channels !\n"); \
		argu::ASSERT(inFeatMaps[0][0].rows <= 0 || inFeatMaps[0][0].cols <= 0, \
					 " input feature maps do not have values !\n");	\
	}
	#endif

	// @brief fc layer input feature maps check
	#ifndef FC_INPUT_INIT
	#define FC_INPUT_INIT(inFeatMaps) \
	{ \
		argu::ASSERT(inFeatMaps.empty() == true, \
					 " input feature maps should not be empty !\n"); \
	}
	#endif


	// @brief non-fc layer input feature maps check
	#ifndef NONFC_OUTPUT_INIT
	#define NONFC_OUTPUT_INIT(ouFeatMaps) \
	{ \
		argu::ASSERT(ouFeatMaps.empty() == true, \
				     " output feature maps should not be empty !\n"); \
		argu::ASSERT(ouFeatMaps[0].empty() == true, \
					 " output feature maps do not have legal channels !\n"); \
		argu::ASSERT(ouFeatMaps[0][0].rows <= 0 || inFeatMaps[0][0].cols <= 0, \
					 " output feature maps do not have values !\n");	\
	}
	#endif

	// @brief fc layer input feature maps check
	#ifndef FC_OUTPUT_INIT
	#define FC_OUTPUT_INIT(ouFeatMaps) \
	{ \
		argu::ASSERT(ouFeatMaps.empty() == true, \
					 " output feature maps should not be empty !\n"); \
	}
	#endif

	// @brief check nnets is well defined
	#ifndef NNETS_INIT
	#define NNETS_INIT(nodeFunc, nodeName) \
	{ \
		if (nodeFunc.empty() || nodeName.empty()) \
			argu::ASSERT(true, " No valid layer in NNets \n"); \
	}
	#endif

	// @brief check mask is well define
	#ifndef NONFC_MASK_INIT
	#define NONFC_MASK_INIT(mask) \
	{ \
		argu::ASSERT(mask.empty() == true, \
					 " mask should not be empty !\n"); \
		argu::ASSERT(mask[0].empty() == true, \
					 " mask do not have legal channels !\n"); \
		argu::ASSERT(mask[0][0].rows <= 0 || inFeatMaps[0][0].cols <= 0, \
					 " mask do not have values !\n");	\
	}
	#endif

	#ifndef FC_MASK_INIT
	#define FC_MASK_INIT(mask) \
	{ \
		argu::ASSERT(mask.empty() == true, \
					 " mask should not be empty !\n"); \
	}
	#endif
}


#endif // _convnet_utility_check_h_