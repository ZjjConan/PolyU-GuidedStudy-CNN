/**
 */

#include "../IMDB/mnist.h"
#include "../Utility/check.h"
#include "../CNN/nnets.h"

#include <ctime>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>


using namespace convnet;
using namespace std;


void loadMNISTData(Mat3D &trainImages, Mat3D &validImages,
				   Mat &trainLabels, Mat &validLabels)
{
	const string trainImageFile = "Data/MNIST/train-images.idx3-ubyte";
	const string trainLabelFile = "Data/MNIST/train-labels.idx1-ubyte";
	const string validImageFile = "data/MNIST/t10k-images.idx3-ubyte";
	const string validLabelFile = "data/MNIST/t10k-labels.idx1-ubyte";

	imdb::MNIST mnistobj;
	mnistobj.loadImages(trainImages, trainImageFile);
	mnistobj.loadImages(validImages, validImageFile);
	mnistobj.loadLabels(trainLabels, trainLabelFile);
	mnistobj.loadLabels(validLabels, validLabelFile);
}


void convertToSingle(Mat3D &images)
{
	for (int i = 0; i < images.size(); ++i)
		images[i].convertTo(images[i], CV_32FC1);
}

void computeDataMean(Mat &meanImage, const Mat3D &images)
{
	for (int i = 0; i < images.size(); ++i)
		meanImage += images[i] / images.size();
}

void abstractDataMean(Mat3D &images, const Mat &meanImage)
{
	for (int i = 0; i < images.size(); ++i)
		images[i] -= meanImage;
}


void createLeNetModel(NNets &model, Mat4D &inFeatMaps,
				      Mat &labels, const int numThreads,
				      const bool verbose = true)
{
	WeightGeometry wparams;
	StrideGeometry strides;
	PadGeometry padding;
	LearnGeometry lparams;

	// conv1 layer
	wparams.set(1, 20, 1, 5, 5, 0.01f);
	strides.set(1, 1);
	padding.set(0, 0, 0, 0);
	lparams.set(0.001f, 0.9f, 1.0f, 0.001f, 0.9f, 1.0f, 0.0005f);
	model.createConvLayer(wparams, strides, padding, lparams, false, numThreads);

	// pool1 layer
	wparams.set(-1, -1, -1, 2, 2, -1);
	strides.set(2, 2);
	padding.set(0, 0, 0, 0);
	model.createPoolLayer(wparams, strides, padding, "Max", false, numThreads);

	// conv2 layer
	wparams.set(1, 50, 20, 5, 5, 0.01f);
	strides.set(1, 1);
	padding.set(0, 0, 0, 0);
	lparams.set(0.001f, 0.9f, 1.0f, 0.001f, 0.9f, 1.0f, 0.0005f);
	model.createConvLayer(wparams, strides, padding, lparams, true, numThreads);

	// pool2 layer	
	wparams.set(-1, -1, -1, 2, 2, -1);
	strides.set(2, 2);
	padding.set(0, 0, 0, 0);
	model.createPoolLayer(wparams, strides, padding, "Max", false, numThreads);

	// concat layer
	wparams.set(-1, -1, -1, 4, 4, -1);
	model.createConcatLayer(wparams, numThreads);

	// fc1 layer
	wparams.set(-1, 500, 50 * 4 * 4, -1, -1, 0.01f);
	lparams.set(0.001f, 0.9f, 1.0f, 0.001f, 0.9f, 1.0f, 0.0005f);
	model.createFCNLayer(wparams, lparams, true, numThreads);

	// activation for fc1 layer
	model.createFCActivLayer("ReLU", numThreads);

	// fc2 layer
	wparams.set(-1, 10, 500, -1, -1, 0.01f);
	lparams.set(0.001f, 0.9f, 1.0f, 0.001f, 0.9f, 1.0f, 0.0005f);
	model.createFCNLayer(wparams, lparams, true, numThreads);

	// loss layer
	model.createLossLayer(numThreads);

	// init and build nnets
	model.setInputImages(inFeatMaps);
	model.setInputLabels(labels);
	model.builChains(true);

	if (verbose) {
		printf("Number of Layers: %d \n", model.getNumberLayers());
		printf("Number of params: %d \n", model.getNumberParams());
		printf("Mode size : %2.2f MB\n", model.getModelSize());
	}
}


void getBatchData(Mat4D &batchData, Mat &batchLabel,
				  const Mat3D &data, const Mat &label, 
				  const vector<int> &index, const int batchSize, 
				  const int batchIdx)
{
	int bs = batchIdx * batchSize;
	int be = std::min((batchIdx + 1)*batchSize, (int)data.size());
	int rows = batchData[0][0].rows;
	int cols = batchData[0][0].cols;
	for (int i = bs; i < be; ++i) {
		memcpy(batchData[i - bs][0].data, data[index[i]].data, rows * rows * sizeof(float));
		label.col(index[i]).copyTo(batchLabel.col(i - bs));
	}
}

void randperm(vector<int> &index)
{
	srand((unsigned int)time(NULL));
	random_shuffle(index.begin(), index.end());
}


void predict(int &numCorrect, const Mat &probs, const Mat &labels)
{
	int *prediction = (int *)calloc(2, sizeof(int));
	float *labelPtr = (float *)labels.data;
	numCorrect = 0;
	for (int i = 0; i < probs.rows; ++i) {
		minMaxIdx(probs.row(i), NULL, NULL, NULL, prediction);
		if ((*labelPtr++) == prediction[1])
			numCorrect++;
	}
	if (prediction != NULL) free(prediction);
	prediction = NULL;
	labelPtr = NULL;
}




int main()
{	
	// global params
	const int numThreads = 5;
	const int batchSize = 100;
	const int chns = 1;
	const int epochs = 5;
	

	printf("Loading MNIST images \n");
	Mat3D trainImages, validImages;
	Mat trainLabels, validLabels;
	loadMNISTData(trainImages, validImages, trainLabels, validLabels);

	printf("Computing data mean \n");
	Mat meanImage = Mat::zeros(28, 28, CV_32FC1);
	convertToSingle(trainImages);
	convertToSingle(validImages);
	computeDataMean(meanImage, trainImages);

	printf("Abstracting mean \n");
	abstractDataMean(trainImages, meanImage);
	abstractDataMean(validImages, meanImage);

	vector<int> trainIndex(60000), validIndex(10000);
	for (int i = 0; i < 60000; ++i)
		trainIndex[i] = i;
	for (int i = 0; i < 10000; ++i)
		validIndex[i] = i;


	printf("Allocating spaces \n");
	Mat4D batchImages(batchSize);
	Mat batchLabels(1, batchSize, CV_32FC1);
	for (int i = 0; i < batchSize; ++i) {
		batchImages[i].resize(1);
		batchImages[i][0] = Mat::zeros(28, 28, CV_32FC1);
	}

	printf("Creating CNN Model \n");
	NNets model;
	createLeNetModel(model, batchImages, batchLabels, numThreads, true);
	
	
	int trainNumBatches = trainImages.size() / batchSize;
	for (int i = 0; i < epochs * trainNumBatches + 1; ++i) {
		int bi = i % trainNumBatches;
		if (bi == 0) {
			randperm(trainIndex);

			// validation
			int validNumBatches = validImages.size() / batchSize;
			int numCorrect = 0;
			for (int vi = 0; vi < validNumBatches; ++vi) {
				double validBatchTime = (double)cv::getTickCount();
				
				getBatchData(batchImages, batchLabels, validImages, validLabels, 
					         validIndex, batchSize, i % validNumBatches);

				// only fprop
				model.fprop();
				float valObjCost = model.getCurrObjCost();
				// compute corrects
				int n;
				predict(n, model.getLayerNode(model.getNumberLayers() - 1)->getFCOuFeatMaps(), batchLabels);
				numCorrect += n;

				validBatchTime = ((double)cv::getTickCount() - validBatchTime) / cv::getTickFrequency();
				printf("Validation batch %d / %d obj %.2f top1e %.4f speed %.3f / s\n", vi % validNumBatches + 1, 
					   validNumBatches, valObjCost, 1 - (float)numCorrect / validIndex.size(),
					   (float)batchSize / validBatchTime);
			}

			if (i == epochs * trainNumBatches) break;
		}

		// training
		double trainBatchTime = (double)cv::getTickCount();
		getBatchData(batchImages, batchLabels, trainImages, trainLabels, 
					 trainIndex, batchSize, bi);

		// forward
		model.fprop();

		float trainObjCost = model.getCurrObjCost();

		// backward
		model.bprop();

		// update
		model.update();

		trainBatchTime = ((double)cv::getTickCount() - trainBatchTime) / cv::getTickFrequency();

		printf("Epochs %d process batch %d / %d obj %.4f speed %.2f data / s\n",
			   i / trainNumBatches + 1, i % trainNumBatches + 1, 
			   trainNumBatches, trainObjCost, (float)batchSize / trainBatchTime);
	}
	
	// release model
	model.release();

	return 0;
}