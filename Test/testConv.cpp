#include <iostream>
#include <opencv2/opencv.hpp>

const int numKernels = 256;
const int numChns = 96;
const int kw = 5;
const int kh = 5;
const int numImages = 100;
const int iw = 55;
const int ih = 55;


static void conv1D(float *res, float const *kernels, float const *image, const int r, const int c, const int l)
{
	for (long int i = 0; i < r*c*l / 4; ++i) {
		*res += kernels[i] + image[i] + kernels[i+1] + image[i+1] + kernels[i+2] + image[i+2] + kernels[i+3] + image[i+3];
	}
}

static void conv1D_sse(float *res, float const *kernels, float const *image, const int r, const int c, const int l)
{
	__m128 xmm0 = _mm_set1_ps(0.0f);
	__m128 xmm1 = _mm_set1_ps(0.0f);
 	__m128 xmm2 = _mm_set1_ps(0.0f);
 	__m128 xmm3 = _mm_set1_ps(0.0f);
	__m128 xmm8 = _mm_set1_ps(0.0f);

	for (long int i = 0; i < r*c*l / 8; i++) {
		xmm0 = _mm_load_ps(kernels + 8 * i);
		xmm1 = _mm_load_ps(image + 8 * i);
		xmm2 = _mm_load_ps(kernels + 8 * i + 4);
		xmm3 = _mm_load_ps(image + 8 * i + 4);
		xmm8 = _mm_add_ps(xmm8, _mm_add_ps(_mm_mul_ps(xmm2, xmm3), _mm_mul_ps(xmm0, xmm1)));
	}

	xmm0 = _mm_hadd_ps(xmm8, xmm8);
	xmm1 = _mm_hadd_ps(xmm0, xmm0);
	*res = _mm_cvtss_f32(xmm1);
}

int main()
{
	float *kernels = (float *)calloc(numChns * numKernels * kw * kh, sizeof(float));
	float *images = (float *)calloc(numChns * numImages * iw * ih, sizeof(float));
	float *results = (float *)calloc(numKernels * 51 * 51 * numImages, sizeof(float));
	for (int i = 0; i < numChns * numKernels * kw * kh; ++i) {
		kernels[i] = (float)rand() / RAND_MAX;
	}
	for (int i = 0; i < numImages * numChns; ++i) {
		images[i] = (float)rand() / RAND_MAX;
	}

	int xstep = 1;
	int ystep = 1;
	int n = 0;
	
	double extime = (double)cv::getTickCount();
	#pragma omp parallel for
	for (int i = 0; i < 100; ++i){
		for (int k = 0; k < numKernels; ++k) {
			for (int r = 0; r < 51; ++r) {
				int roff = r * numChns * iw;
				for (int c = 0; c < 51; ++c) {
					int coff = c * numChns;
					conv1D_sse(results + n, kernels, images + roff + coff, kh, kw, numChns);
					n++;
				}
			}
		}
	}
	printf("conv time %fs\n", ((double)cv::getTickCount() - extime) / cv::getTickFrequency());
	
}