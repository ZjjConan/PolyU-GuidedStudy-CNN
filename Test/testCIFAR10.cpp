/**
*/

#include "../IMDB/cifar.h"
#include "../CNN/nnets.h"
#include <ctime>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>


using namespace convnet;
using namespace std;

void addTo(Mat4D &images, Mat &labels, const Mat4D &batchImages, 
		   const Mat &batchLabels, const int batchIndex)
{
	const int numImages = 10000;
	const int rows = 32;
	const int cols = 32;
	const int chns = 3;

	float *labelPtr = (float *)labels.data;
	float *bLabelPtr = (float *)batchLabels.data;
	int bo = batchIndex * numImages;
	for (int i = 0; i < numImages; ++i, ++bo) {
		batchImages[i][0].convertTo(images[bo][0], CV_32FC1);
		batchImages[i][1].convertTo(images[bo][1], CV_32FC1);
		batchImages[i][2].convertTo(images[bo][2], CV_32FC1);
		*labelPtr++ = *bLabelPtr++;
	}
}

void loadCIFARData(Mat4D &trainImages, Mat4D &validImages,
				   Mat &trainLabels, Mat &validLabels)
{
	const string trainImageBatchFile1 = "Data/cifar-10-batches-bin/data_batch_1.bin";
	const string trainImageBatchFile2 = "Data/cifar-10-batches-bin/data_batch_2.bin";
	const string trainImageBatchFile3 = "Data/cifar-10-batches-bin/data_batch_3.bin";
	const string trainImageBatchFile4 = "Data/cifar-10-batches-bin/data_batch_4.bin";
	const string trainImageBatchFile5 = "Data/cifar-10-batches-bin/data_batch_5.bin";
	const string validImageBatchFile = "Data/cifar-10-batches-bin/test_batch.bin";

	imdb::CIFAR10 cifarobj;
	Mat4D batchImages;
	Mat batchLabels;

	printf("Loading training batch 1 \n");
	cifarobj.loadBatch(batchImages, batchLabels, trainImageBatchFile1);
	addTo(trainImages, trainLabels.colRange(0, 10000), batchImages, batchLabels, 0);

	printf("Loading training batch 2 \n");
	cifarobj.loadBatch(batchImages, batchLabels, trainImageBatchFile2);
	addTo(trainImages, trainLabels.colRange(10000, 20000), batchImages, batchLabels, 1);

	printf("Loading training batch 3 \n");
	cifarobj.loadBatch(batchImages, batchLabels, trainImageBatchFile3);
	addTo(trainImages, trainLabels.colRange(20000, 30000), batchImages, batchLabels, 2);

	printf("Loading training batch 4 \n");
	cifarobj.loadBatch(batchImages, batchLabels, trainImageBatchFile4);
	addTo(trainImages, trainLabels.colRange(30000, 40000), batchImages, batchLabels, 3);

	printf("Loading training batch 5 \n");
	cifarobj.loadBatch(batchImages, batchLabels, trainImageBatchFile5);
	addTo(trainImages, trainLabels.colRange(40000, 50000), batchImages, batchLabels, 4);

	printf("Loading validation batch \n");
	cifarobj.loadBatch(batchImages, batchLabels, validImageBatchFile);
	addTo(validImages, validLabels, batchImages, batchLabels, 0);
}


void convertToSingle(Mat4D &images)
{
	for (int i = 0; i < images.size(); ++i) {
		images[i][0].convertTo(images[i][0], CV_32FC1, 1.0f / 255);
		images[i][1].convertTo(images[i][1], CV_32FC1, 1.0f / 255);
		images[i][2].convertTo(images[i][2], CV_32FC1, 1.0f / 255);
	}
}

void computeDataMean(Mat3D &meanImage, const Mat4D &images)
{
	for (int i = 0; i < images.size(); ++i) {
		meanImage[0] += images[i][0] / images.size();
		meanImage[1] += images[i][1] / images.size();
		meanImage[2] += images[i][2] / images.size();
	}
}

void abstractDataMean(Mat4D &images, const Mat3D &meanImage)
{
	for (int i = 0; i < images.size(); ++i) {
		images[i][0] -= meanImage[0];
		images[i][1] -= meanImage[1];
		images[i][2] -= meanImage[2];
	}
}

void contrastNormalize(Mat4D &images)
{
	Mat mean, std;
	for (int i = 0; i < images.size(); ++i) {
		for (int ch = 0; ch < images[0].size(); ++ch) {
			cv::meanStdDev(images[0][ch], mean, std);
			images[0][ch] = images[0][ch] - mean;
			images[0][ch] = images[0][ch] / std;
		}
	}
}

void createFastCNNModel(NNets &model, Mat4D &inFeatMaps, 
					    Mat &labels, const int numThreads,
						const bool verbose = true)
{
	WeightGeometry wparams;
	StrideGeometry strides;
	PadGeometry padding;
	LearnGeometry lparams;

	// conv1 layer
	wparams.set(1, 32, 3, 5, 5, 0.0001f);
	strides.set(1, 1);
	padding.set(2, 2, 2, 2);
	lparams.set(0.002f, 0.9f, 0.1f, 0.001f, 0.9f, 0.1f, 0.004f);
	model.createConvLayer(wparams, strides, padding, lparams, false, numThreads);

	// pool1 layer
	wparams.set(-1, -1, -1, 3, 3, -1);
	strides.set(2, 2);
	padding.set(0, 0, 0, 0);
	model.createPoolLayer(wparams, strides, padding, "Max", false, numThreads);

	// relu1 layer
	model.createActivLayer("ReLU", numThreads);

	// conv2 layer
	wparams.set(1, 32, 32, 5, 5, 0.01f);
	strides.set(1, 1);
	padding.set(2, 2, 2, 2);
	lparams.set(0.002f, 0.9f, 0.1f, 0.001f, 0.9f, 0.1f, 0.004f);
	model.createConvLayer(wparams, strides, padding, lparams, true, numThreads);

	// relu2 layer
	model.createActivLayer("ReLU", numThreads);

	// pool2 layer	
	wparams.set(-1, -1, -1, 3, 3, -1);
	strides.set(2, 2);
	padding.set(0, 0, 0, 0);
	model.createPoolLayer(wparams, strides, padding, "Avg", true, numThreads);

	// conv3 layer
	wparams.set(1, 64, 32, 5, 5, 0.01f);
	strides.set(1, 1);
	padding.set(2, 2, 2, 2);
	lparams.set(0.002f, 0.9f, 0.1f, 0.001f, 0.9f, 0.1f, 0.004f);
	model.createConvLayer(wparams, strides, padding, lparams, true, numThreads);

	// relu3 layer
	model.createActivLayer("ReLU", numThreads);

	// pool3 layer	
	wparams.set(-1, -1, -1, 3, 3, -1);
	strides.set(2, 2);
	padding.set(0, 0, 0, 0);
	model.createPoolLayer(wparams, strides, padding, "Avg", true, numThreads);

	// concat layer
	wparams.set(-1, -1, -1, 3, 3, -1);
	model.createConcatLayer(wparams, numThreads);

	// fc4 layer
	wparams.set(-1, 64, 64 * 3 * 3, -1, -1, 0.1f);
	lparams.set(0.002f, 0.9f, 0.1f, 0.001f, 0.9f, 0.1f, 0.03f);
	model.createFCNLayer(wparams, lparams, true, numThreads);

	// activation for fc4 layer
	model.createFCActivLayer("ReLU", numThreads);

	// fc5 layer
	wparams.set(-1, 10, 64, -1, -1, 0.1f);
	lparams.set(0.002f, 0.9f, 0.1f, 0.001f, 0.9f, 0.1f, 0.03f);
	model.createFCNLayer(wparams, lparams, true, numThreads);

	// loss layer
	model.createLossLayer(numThreads);

	model.setInputImages(inFeatMaps);
	model.setInputLabels(labels);
	model.builChains(true);

	if (verbose) {
		printf("Number of Layers: %d \n", model.getNumberLayers());
		printf("Number of params: %d \n", model.getNumberParams());
		printf("Mode size : %2.2f MB \n", model.getModelSize());
	}
}


void createCompCNNModel(NNets &model, Mat4D &inFeatMaps, 
						Mat &labels, const int numThreads, 
						const bool verbose = true)
{
	WeightGeometry wparams;
	StrideGeometry strides;
	PadGeometry padding;
	LearnGeometry lparams;

	// conv1 layer
	wparams.set(1, 32, 3, 5, 5, 0.0001f);
	strides.set(1, 1);
	padding.set(2, 2, 2, 2);
	lparams.set(0.002f, 0.9f, 0.1f, 0.001f, 0.9f, 0.1f, 0.004f);
	model.createConvLayer(wparams, strides, padding, lparams, false, numThreads);

	// pool1 layer
	wparams.set(-1, -1, -1, 3, 3, -1);
	strides.set(2, 2);
	padding.set(0, 0, 0, 0);
	model.createPoolLayer(wparams, strides, padding, "Max", false, numThreads);

	// relu1 layer
	model.createActivLayer("ReLU", numThreads);

	// conv2 layer
	wparams.set(1, 32, 32, 5, 5, 0.01f);
	strides.set(1, 1);
	padding.set(2, 2, 2, 2);
	lparams.set(0.002f, 0.9f, 0.1f, 0.001f, 0.9f, 0.1f, 0.004f);
	model.createConvLayer(wparams, strides, padding, lparams, true, numThreads);

	// relu2 layer
	model.createActivLayer("ReLU", numThreads);

	// pool2 layer	
	wparams.set(-1, -1, -1, 3, 3, -1);
	strides.set(2, 2);
	padding.set(0, 0, 0, 0);
	model.createPoolLayer(wparams, strides, padding, "Avg", true, numThreads);

	// conv3 layer
	wparams.set(1, 64, 32, 5, 5, 0.01f);
	strides.set(1, 1);
	padding.set(2, 2, 2, 2);
	lparams.set(0.002f, 0.9f, 0.1f, 0.001f, 0.9f, 0.1f, 0.004f);
	model.createConvLayer(wparams, strides, padding, lparams, true, numThreads);

	// relu3 layer
	model.createActivLayer("ReLU", numThreads);

	// pool3 layer	
	wparams.set(-1, -1, -1, 3, 3, -1);
	strides.set(2, 2);
	padding.set(0, 0, 0, 0);
	model.createPoolLayer(wparams, strides, padding, "Avg", true, numThreads);

	// concat layer
	wparams.set(-1, -1, -1, 3, 3, -1);
	model.createConcatLayer(wparams, numThreads);

	// fc4 layer
	wparams.set(-1, 64, 64 * 3 * 3, -1, -1, 0.1f);
	lparams.set(0.002f, 0.9f, 0.1f, 0.001f, 0.9f, 0.1f, 1.0f);
	model.createFCNLayer(wparams, lparams, true, numThreads);

	// activation for fc4 layer
	model.createFCActivLayer("ReLU", numThreads);

	// fc5 layer
	wparams.set(-1, 10, 64, -1, -1, 0.1f);
	lparams.set(0.002f, 0.9f, 0.1f, 0.001f, 0.9f, 0.1f, 1.0f);
	model.createFCNLayer(wparams, lparams, true, numThreads);

	// loss layer
	model.createLossLayer(numThreads);

	model.setInputImages(inFeatMaps);
	model.setInputLabels(labels);
	model.builChains(true);

	if (verbose) {
		printf("Number of Layers: %d \n", model.getNumberLayers());
		printf("Number of params: %d \n", model.getNumberParams());
		printf("Mode size : %2.2f MB \n", model.getModelSize());
	}
}



void getBatchData(Mat4D &batchData, Mat &batchLabel,
			      const Mat4D &data, const Mat &label, const vector<int> &index,
				  const int batchSize, const int batchIdx)
{
	int bs = batchIdx * batchSize;
	int be = min((batchIdx + 1)*batchSize, (int)data.size());
	int rows = batchData[0][0].rows;
	int cols = batchData[0][0].cols;
	for (int i = bs; i < be; ++i) {
		for (int ch = 0; ch < 3; ++ch)
			memcpy(batchData[i - bs][ch].data, data[index[i]][ch].data, rows * cols * sizeof(float));
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
	int *predPtr = (int *)calloc(2, sizeof(int));
	float *labelPtr = (float *)labels.data;
	numCorrect = 0;
	for (int i = 0; i < probs.rows; ++i) {
		minMaxIdx(probs.row(i), NULL, NULL, NULL, predPtr);
		if ((*labelPtr++) == predPtr[1])
			numCorrect++;
	}
	if (predPtr != NULL) free(predPtr);
	predPtr = NULL;
	labelPtr = NULL;
}


int main()
{
	Mat4D trainImages(50000);
	Mat4D validImages(10000);
	Mat trainLabels(1, 50000, CV_32FC1);
	Mat validLabels(1, 10000, CV_32FC1);
	
	printf("Allocating training and validation spaces \n");
	for (int i = 0; i < 50000; ++i) {
		trainImages[i].resize(3);
		trainImages[i][0] = Mat::zeros(32, 32, CV_32FC1);
		trainImages[i][1] = Mat::zeros(32, 32, CV_32FC1);
		trainImages[i][2] = Mat::zeros(32, 32, CV_32FC1);
	}

	for (int i = 0; i < 10000; ++i) {
		validImages[i].resize(3);
		validImages[i][0] = Mat::zeros(32, 32, CV_32FC1);
		validImages[i][1] = Mat::zeros(32, 32, CV_32FC1);
		validImages[i][2] = Mat::zeros(32, 32, CV_32FC1);
	}

	printf("Loading CIFAR data \n");
	loadCIFARData(trainImages, validImages, trainLabels, validLabels);

	Mat3D meanImage(3);
	meanImage[0] = Mat::zeros(32, 32, CV_32FC1);
	meanImage[1] = Mat::zeros(32, 32, CV_32FC1);
	meanImage[2] = Mat::zeros(32, 32, CV_32FC1);
	printf("Computing data mean \n");
	//convertToSingle(trainImages);
	//convertToSingle(validImages);
	computeDataMean(meanImage, trainImages);

	printf("Abstracting data mean \n");
	abstractDataMean(trainImages, meanImage);
	abstractDataMean(validImages, meanImage);


	// --------------------------------------------------------------------------
	//
	//								cifar fast model
	// chains:
	//		x -> conv1 -> max pool1 -> relu1 -> conv2 -> relu2 -> avg pool2 
	//		  -> conv3 -> relu3 -> avg pool3 -> concat -> fc4 -> fcActiv4 
	//		  -> fc5 -> softmaxloss
	//
	//
	// --------------------------------------------------------------------------
	printf("Creating CNN model \n");
	
	const int numThreads = 5;
	const int epochs = 10;
	const int batchSize = 100;

	Mat4D batchImages(batchSize);
	Mat batchLabels(1, batchSize, CV_32FC1);
	for (int i = 0; i < batchSize; ++i) {
		batchImages[i].resize(3);
		batchImages[i][0] = Mat::zeros(32, 32, CV_32FC1);
		batchImages[i][1] = Mat::zeros(32, 32, CV_32FC1);
		batchImages[i][2] = Mat::zeros(32, 32, CV_32FC1);
	}

	NNets model;
	createFastCNNModel(model, batchImages, batchLabels, numThreads, true);
	
	vector<int> trainIndex(trainImages.size());
	vector<int> validIndex(validImages.size());
	for (int i = 0; i < trainIndex.size(); ++i) {
		trainIndex[i] = i;
	}
	
	for (int i = 0; i < validIndex.size(); ++i) {
		validIndex[i] = i;
	}
	
	// --------------------------------------------------------------------------
	// 
	//									training 
	// 
	// --------------------------------------------------------------------------	
	int n = 0;
	int numTrainBatches = trainImages.size() / batchSize;
	for (int i = 0; i < epochs * numTrainBatches + 1; ++i) {
		int bi = i % numTrainBatches;
		if (bi == 0) {
			randperm(trainIndex);
			n++;
			if (n == 8) {
				model.scaleLearningRate();
				printf("Scale learning rate \n");
			}

			// validation
			int numValidBatches = validImages.size() / batchSize;
			int numCorrect = 0;
			for (int vi = 0; vi < numValidBatches; ++vi) {
				double valbatchTime = (double)cv::getTickCount();
				getBatchData(batchImages, batchLabels, validImages,
					         validLabels, validIndex, batchSize, vi);

				model.fprop();
				float valObjCost = model.getCurrObjCost();
				int n;
				predict(n, model.getLayerNode(model.getNumberLayers() - 1)->getFCOuFeatMaps(), batchLabels);
				numCorrect += n;

				valbatchTime = ((double)cv::getTickCount() - valbatchTime) / cv::getTickFrequency();
				printf("Validation process batch %d / %d obj %.4f top1e %.3f speed %.2f/s\n",
					    vi % numValidBatches + 1, numValidBatches, valObjCost, 
						1 - (float)numCorrect / validImages.size(),
						batchSize / (float)valbatchTime);
			}

			if (i == epochs * numTrainBatches) break;
		}


		// training
		double batchTime = (double)cv::getTickCount();
		getBatchData(batchImages, batchLabels, trainImages, 
					 trainLabels, trainIndex, batchSize, bi);
		// forward
		model.fprop();

		float traObjCost = model.getCurrObjCost();

		// backward
		model.bprop();

		// update
		model.update();

		batchTime = ((double)cv::getTickCount() - batchTime) / cv::getTickFrequency();

		printf("Epochs %d obj %.4f process batch %d / %d speed %.2f data / s\n",
			   i / numTrainBatches + 1, traObjCost,
			   i % numTrainBatches + 1, numTrainBatches, (float)batchSize / batchTime);
	}
	return 0;
}