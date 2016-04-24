#ifndef _CONVNET_UTILITY_TYPES_H_
#define _CONVNET_UTILITY_TYPES_H_
#pragma once

#include <vector>
#include <opencv2/core/core.hpp>

namespace convnet
{
	#define Mat4D std::vector<std::vector<cv::Mat>>
	#define Mat3D std::vector<cv::Mat>
}

#endif // types used in cnn