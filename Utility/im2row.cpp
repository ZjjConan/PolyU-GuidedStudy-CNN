
#include "check.h"
#include "im2row.h"
#include <opencv2/core/core.hpp>

namespace convnet
{
	// -----------------------------------------------------------------
	// im2row
	// -----------------------------------------------------------------
	void im2row(Mat &colImage, const vector<Mat> &images,
				const int winHeight, const int winWidth,
				const int stepRow, const int stepCol,
				const int padTop, const int padLeft,
				const int padBottom, const int padRight)
	{
		int imrows = images[0].rows;
		int imcols = images[0].cols;
		int maxrows = images[0].rows + padBottom - winHeight + 1;
		int maxcols = images[0].cols + padRight - winWidth + 1;
		int chns = images.size();
	
		float *imagePtr = NULL;
		float *colImPtr = CV_MAT_PRF(colImage);
		for (int r = -padTop; r < maxrows; r += stepRow) {
			for (int c = -padLeft; c < maxcols; c += stepCol) {	
				for (int ch = 0; ch < chns; ++ch) {
					imagePtr = CV_MAT_PRF(images[ch]);
					for (int br = r; br < r + winHeight; ++br) {
						for (int bc = c; bc < c + winWidth; ++bc) {
							if (br < 0 || br >= imrows || bc < 0 || bc >= imcols)
								*colImPtr++ = 0;
							else
								*colImPtr++ = CV_MAT_AT(imagePtr, br, bc, imcols);
						}
					}
				}
			}
		}
		imagePtr = NULL;
		colImPtr = NULL;
	}

	// -----------------------------------------------------------------
	// row2im
	// -----------------------------------------------------------------
	void row2im(vector<Mat> &images, const Mat &colImage,
			    const int winHeight, const int winWidth,
				const int stepRow, const int stepCol,
				const int padTop, const int padLeft,
				const int padBottom, const int padRight)
	{
		int imrows = images[0].rows;
		int imcols = images[0].cols;
		int maxrows = images[0].rows + padBottom - winHeight + 1;
		int maxcols = images[0].cols + padRight - winWidth + 1;
		int chns = images.size();

		float *imagePtr = NULL;
		float *colImPtr = CV_MAT_PRF(colImage);
		for (int r = -padTop; r < maxrows; r += stepRow) {
			for (int c = -padLeft; c < maxcols; c += stepCol) {
				for (int ch = 0; ch < chns; ++ch) {
					imagePtr = CV_MAT_PRF(images[ch]);
					for (int br = r; br < r + winHeight; ++br) {
						for (int bc = c; bc < c + winWidth; ++bc) {
							if (br < 0 || br >= imrows || bc < 0 || bc >= imcols)
								CV_MAT_AT(imagePtr, br, bc, imcols) = 0;
							else
								CV_MAT_AT(imagePtr, br, bc, imcols) += *colImPtr++;
						}
					}
				}
			}
		}
		imagePtr = NULL;
		colImPtr = NULL;
	}


	// -----------------------------------------------------------------
	// im2col
	// -----------------------------------------------------------------
	void im2col(Mat &colImage, const vector<Mat> &images,
				const int winHeight, const int winWidth,
				const int stepRow, const int stepCol,
				const int padTop, const int padLeft,
			    const int padBottom, const int padRight)
	{
 		int imrows = images[0].rows;
 		int imcols = images[0].cols;
 		int blength = winHeight * winWidth;
		int bdims = blength * images.size();
		
 		int numBlocksPerRow = (imcols + (padRight + padLeft) - winWidth) / stepCol + 1;
		int numBlocksPerCol = (imrows + (padTop + padBottom) - winHeight) / stepRow + 1;
 		float *imagePtr = NULL;
 		float *colImPtr = CV_MAT_PRF(colImage);
		
		for (int d = 0; d < bdims; ++d) {
			int bchns = d / blength;
			int brows = d / winHeight;
			int bcols = d % winWidth;
			brows = brows % winHeight;

			imagePtr = CV_MAT_PRF(images[bchns]);
 			for (int nb = 0; nb < numBlocksPerCol * numBlocksPerRow; ) {
				int roff = nb / numBlocksPerRow;
				int r = brows + roff * stepRow - padTop;
				if (r < 0 || r >= imrows) {
					for (int ni = 0; ni < numBlocksPerRow; ++ni)
						*colImPtr++ = 0;
				}
				else {
					int coff = nb % numBlocksPerRow;
					int c = bcols + coff * stepCol - padLeft;
					int maxc = imcols + padRight - winWidth + bcols + 1;
					int valc = min(imcols, maxc);
					for (; c < 0; c += stepCol)
						*colImPtr++ = 0;

					for (; c < valc; c += stepCol)
						*colImPtr++ = CV_MAT_AT(imagePtr, r, c, imcols);

					for (; c < maxc; c += stepCol)
 						*colImPtr++ = 0;
				}
				nb = nb + numBlocksPerRow;
 			}
		}
		imagePtr = NULL;
		colImPtr = NULL;
	}

	// -----------------------------------------------------------------
	// col2im
	// -----------------------------------------------------------------
	void col2im(vector<Mat> &images, const Mat &colImage, 
			    const int winHeight, const int winWidth, 
				const int stepRow, const int stepCol, 
				const int padTop, const int padLeft, 
				const int padBottom, const int padRight)
	{
		int imrows = images[0].rows;
		int imcols = images[0].cols;
		int blength = winHeight * winWidth;
		int bdims = blength * images.size();

		int numBlocksPerRow = (imcols + (padRight + padLeft) - winWidth) / stepCol + 1;
		int numBlocksPerCol = (imrows + (padTop + padBottom) - winHeight) / stepRow + 1;
		int numBlocks = numBlocksPerRow * numBlocksPerCol;
		float *imagePtr = NULL;
		float *colImPtr = CV_MAT_PRF(colImage);

		for (int d = 0; d < bdims; ++d) {
			int bchns = d / blength;
			int brows = d / winHeight;
			int bcols = d % winWidth;
			brows = brows % winHeight;

			// something wrong here !!! warning
			imagePtr = CV_MAT_PRF(images[bchns]);
			for (int nb = 0; nb < numBlocksPerCol * numBlocksPerRow; ++nb) {
				int roff = nb / numBlocksPerRow;
				int r = brows + roff * stepRow - padTop;
				if (r >= 0 && r < imrows) {
					int coff = nb % numBlocksPerRow;
					int c = bcols + coff * stepCol - padLeft;
					if (c >= 0 && c < imcols) {
						int imc = imcols - winWidth + bcols + 1;
						for (; c < imc; c += stepCol, ++nb)
							CV_MAT_AT(imagePtr, r, c, imcols) += CV_MAT_AT(colImPtr, d, nb, numBlocks);
					}
				}
			}
		}
		imagePtr = NULL;
		colImPtr = NULL;

// 		int width = images[0].cols;
// 		int height = images[0].rows;
// 		int depth = images.size();
// 
// 		int numPatchesX = (width + (padLeft + padRight) - winWidth) / stepCol + 1;
// 		int numPatchesY = (height + (padTop + padBottom) - winHeight) / stepRow + 1;
// 		int numRows = winWidth * winHeight * depth;
// 
// 		
// 		float *stacked = CV_MAT_PRF(colImage);
// 		memset(images[0].data, 0, sizeof(float)* width * height);
// 		memset(images[1].data, 0, sizeof(float)* width * height);
// 		/*
// 		Do the converse of im2col, still scanning rows of the stacked image.
// 		See comments of im2col for an explanation of the algorithm.
// 		*/
// 		for (int row = 0; row < numRows; ++row) {
// 			int u = row;
// 			int v = u / winWidth;
// 			int z = v / winHeight;
// 			u %= winWidth;
// 			v %= winHeight;
// 
// 			int x0 = static_min(numPatchesX, ceil_divide(padLeft - u, stepCol));
// 			int y0 = static_min(numPatchesY, ceil_divide(padTop - v, stepRow));
// 			int x1 = static_min(numPatchesX, floor_divide(width - 1 + padLeft - u, stepCol) + 1);
// 			int y1 = static_min(numPatchesY, floor_divide(height - 1 + padTop - v, stepRow) + 1);
// 			int x;
// 			int y;
// 
// 			y = static_max(0, y0);
// 			stacked += numPatchesX * static_max(y, 0);
// 			for (; y < y1; ++y) {
// 				x = static_max(0, x0);
// 				int y_data = y * stepRow + v - padTop;
// 				int x_data = x * stepCol + u - padLeft;
// 				float * b = (float *)images[u].data + (z * height + y_data) * width + x_data;
// 				stacked += x;
// 				for (; x < x1; ++x) {
// 					*b += *stacked++;
// 					b += stepCol;
// 				}
// 				stacked += numPatchesX - x;
// 			}
// 			stacked += numPatchesX * (numPatchesY - y);
// 		}
	}
}