#ifndef _CONVNET_UTILITY_MATMUL_H_
#define _CONVNET_UTILITY_MATMUL_H_
#pragma once

namespace convnet
{
	void mmul(float *dst, const float *mat1, const float *mat2,
  			  const int rows, const int dims, const int cols);
// 
// 
// 	void mmula(float *dst, const float *mat1, const float *mat2,
// 			   const int rows, const int dims, const int cols);

	void fastMatMul(float *Z, const float *X, const float *Y,
				    const int xrows, const int xcols,
					const int yrows, const int ycols,
					const bool trans1 = false, 
					const bool trans2 = false);

	void fastMatMulAdd(float *Z, const float *X, const float *Y,
					   const int xrows, const int xcols,
					   const int yrows, const int ycols,
					   const bool trans1 = false,
					   const bool trans2 = false);
}


#endif // matmul.h