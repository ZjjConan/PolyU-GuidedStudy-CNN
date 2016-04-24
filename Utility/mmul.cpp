#include "check.h"
#include "mmul.h"
#include <opencv2/core/core.hpp>
#include <intrin.h>
#include <arrayfire.h>
#include <stdlib.h>

namespace convnet
{
	using namespace std;
	using namespace cv;

	float vdot(const float *x, const float *y, const int d)
	{	
		float res(0.0f);
		for (int i = 0; i < d; ++i)
			res += x[i] * y[i];
		return res;
	}


	// forms the dot product of two vectors.
	// uses unrolled loops for increments equal to one.
	// jack dongarra, linpack, 3/11/78.
	// modified 12/3/93, array(1) declarations changed to array(*)
	//	
	// the original function is ddot_ for double dot product of two vectors
	// modified by Lx.Yang for float dot product
	float vdotunroll(const float *x, const float *y, const int d)
	{
		long int i, m;
		
 		m = d - 4;
 		float res = 0.0f;
 		//! for loop unrolling, nearly twice faster
 		for (i = 0; i < m; i += 5){
 			res += x[i] * y[i] + x[i+1] * y[i+1] + x[i+2] * y[i+2] + 
 				   x[i+3] * y[i+3] + x[i+4] * y[i+4];
 		}
 		for (; i < d; ++i) {
 			res += x[i] * y[i];
 		}
 
 		return res;
	}

 	float vdotsse(const float *x, const float *y, const int d)
 	{
 		if (d < 8) {
 			long int i, m;
 			m = d - 4;
 			float res = 0.0f;
 			// for loop unrolling, nearly twice faster
 			for (i = 0; i < m; i += 5){
 				res += x[i] * y[i] + x[i + 1] * y[i + 1] + x[i + 2] * y[i + 2] +
 					   x[i + 3] * y[i + 3] + x[i + 4] * y[i + 4];
 			}
 			for (; i < d; ++i) {
 				res += x[i] * y[i];
 			}
 
 			return res;
 		}
 
 		__m128 xmm0 = _mm_set1_ps(0.0f);
 		__m128 xmm1 = _mm_set1_ps(0.0f);
 		__m128 xmm2 = _mm_set1_ps(0.0f);
 		__m128 xmm3 = _mm_set1_ps(0.0f);
 		__m128 xmm8 = _mm_set1_ps(0.0f);
 
 		long int i, m;
 		m = d / 8;
 		for (i = 0; i < m; i++) {
 			xmm0 = _mm_load_ps(x + 8 * i);
 			xmm1 = _mm_load_ps(y + 8 * i);
 			xmm2 = _mm_load_ps(x + 8 * i + 4);
 			xmm3 = _mm_load_ps(y + 8 * i + 4);
 			xmm8 = _mm_add_ps(xmm8, _mm_add_ps(_mm_mul_ps(xmm2, xmm3), _mm_mul_ps(xmm0, xmm1)));
 		}
		
		xmm0 = _mm_hadd_ps(xmm8, xmm8);
		xmm1 = _mm_hadd_ps(xmm0, xmm0);
		return _mm_cvtss_f32(xmm1);
 	}
 
 
 	// much faster than opencv matrix-multiplication
 	void mmul(float *dst, const float *mat1, const float *mat2,
 			  const int rows, const int dims, const int cols)
 	{
 		for (int r = 0; r < rows; ++r) {
 			int ro = r * dims;
			for (int c = 0; c < cols; ++c) {
 				int co = c * dims;
				dst[r * cols + c] = vdotsse(mat1 + ro, mat2 + co, dims);
 			}
 		}
 	}
 
// 	void mmula(float *dst, const float *mat1, const float *mat2,
// 			   const int rows, const int dims, const int cols)
// 	{
// 		for (int r = 0; r < rows; ++r) {
// 			int ro = r * dims;
// 			for (int c = 0; c < cols; ++c) {
// 				int co = c * dims;
// 				dst[r * cols + c] += vdotsse(mat1 + ro, mat2 + co, dims);
// 			}
// 		}
// 	}


	void fastMatMul(float *Z, const float *X, const float *Y,
					const int xrows, const int xcols, 
					const int yrows, const int ycols,
					const bool trans1, const bool trans2)
	{
		af::array fastmat1(xcols, xrows, X, afHost);
		af::array fastmat2(ycols, yrows, Y, afHost);
		af::array fastdst;
		float *res = NULL;
		if (!trans1 && !trans2) {
			fastdst = af::matmul(fastmat2, fastmat1);
			res = fastdst.host<float>();
			memcpy(Z, res, ycols * xrows * sizeof(float));
		}
		else if (trans1 && !trans2) {
			fastdst = af::matmulNT(fastmat2, fastmat1);
			res = fastdst.host<float>();
			memcpy(Z, res, ycols * xcols * sizeof(float));
		}
		else if (!trans1 && trans2) {
			fastdst = af::matmulTN(fastmat2, fastmat1);
			res = fastdst.host<float>();
			memcpy(Z, res, yrows * xrows * sizeof(float));
		}
		else if (trans1 && trans2) {
			fastdst = af::matmul(fastmat1, fastmat2);
			res = fastdst.host<float>();
			memcpy(Z, res, xcols * yrows * sizeof(float));
		}
	#ifndef _DEBUG
		if (res != NULL) delete[] res;
	#endif
	}


	void fastMatMulAdd(float *Z, const float *X, const float *Y,
					   const int xrows, const int xcols,
					   const int yrows, const int ycols,
					   const bool trans1, const bool trans2)
	{
		af::array fastmat1(xcols, xrows, X);
		af::array fastmat2(ycols, yrows, Y);
		af::array fastdst;
		int dstrows, dstcols;
		if (!trans1 && !trans2) {
			fastdst = af::matmul(fastmat2, fastmat1);
			dstrows = ycols;
			dstcols = xrows;
		}
		else if (trans1 && !trans2) {
			fastdst = af::matmulNT(fastmat2, fastmat1);
			dstrows = ycols;
			dstcols = xcols;
		}
		else if (!trans1 && trans2) {
			fastdst = af::matmulTN(fastmat2, fastmat1);
			dstrows = yrows;
			dstcols = xrows;
		}
		else if (trans1 && trans2) {
			fastdst = af::matmul(fastmat1, fastmat2);
			dstrows = xcols;
			dstcols = yrows;
		}

		float *res = fastdst.host<float>();
		for (int i = 0; i < dstrows * dstcols; ++i) {
			Z[i] += res[i];
		}
	#ifndef _DEBUG
		if (res != NULL) delete[] res;
	#endif
	}
}