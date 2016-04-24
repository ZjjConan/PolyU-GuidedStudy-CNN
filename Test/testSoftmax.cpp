/**
*/

#include "../IMDB/mnist.h"
#include "../CNN/layer.h"
#include "../CNN/loss.h"
#include "../CNN/fcLayer.h"
#include "../CNN/concatLayer.h"

#include <ctime>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace convnet;


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
	const int epochs = 10;


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
	
	ConcatLayer node1(batchImages);
	node1.init();
	FCLayer node2(node1.getFCOuFeatMaps());
	node2.setWeightGeometry(10, 28 * 28, 0.01);
	node2.setUpdaterParams(0.1, 0.95, 0.1, 0.1, 0.95, 0.1, 0.001);
	node2.init();
	SoftmaxLoss softmax(node2.getFCOuFeatMaps(), batchImages);
	softmax.init();

	// training
	vector<int> index(trainImages.size());
	for (int i = 0; i < trainImages.size(); ++i) {
		index[i] = i;
	}
	
	int numBatches = trainImages.size() / batchSize;
 	double exttime = (double)cv::getTickCount();
 	for (int it = 0; it < 5 * numBatches; ++it) {
 		int bi = it % numBatches;
 		if (bi == 0) {
 			randperm(index);
 			if (it != 0)
 				node2.scaleLearningRate();
 		}
 
 		getBatchData(batchImages, batchLabels, trainImages, trainLabels, index, batchSize, bi);
		
		node1.fprop();
		node2.fprop();
 		softmax.fprop();
 		
 		float objCost = softmax.getCurrObjCost();
 		objCost += node1.getCurrObjCost();
		objCost += node2.getCurrObjCost();

 		softmax.bprop();
 		node2.bprop();
		node1.bprop();

 		node1.update();
 		printf("iter: %d current cost %.4f\n", it, objCost);
 	}
 	printf("training time %.2fs\n", ((double)cv::getTickCount() - exttime) / getTickFrequency());
  	
	// prediction for training
	float trainAccuracy = 0;
	for (int i = 0; i < numBatches; ++i) {
		int bi = i % numBatches;
		getBatchData(tmData, batchLabel, digits, labels, index, batchSize, bi); 	
		tmData.convertTo(batchData, CV_32FC1, 1.0 / 255);
		fullyConnect.fprop();
		softmax.fprop();
		trainAccuracy += (float)softmax.g();
	}
	printf("training accuracy %4.2f%%\n", 100 * trainAccuracy / digits.size());
 	
 	digits.clear();
 	labels.release();
	
	imdb.loadDigits(testImageFile, digits);
	imdb.loadLabels(testLabelFile, labels);
	for (int i = 0; i < digits.size(); ++i) {
		index[i] = i;
	}
		
	float testAccuracy = 0;
	numBatches = digits.size() / batchSize;
	for (int i = 0; i < numBatches; ++i) {
		getBatchData(tmData, batchLabel, digits, labels, index, batchSize, i);
		tmData.convertTo(batchData, CV_32FC1, 1.0 / 255);
		fullyConnect.fprop();
		softmax.fprop();
		testAccuracy += (float)softmax.getCorrectNum();
	}
	printf("testing accuracy %4.2f%%\n", 100 * testAccuracy / digits.size());


// 		opts::im2single(batchData, tempData);
// 		softmax.updateInMapsRef(batchData);
// 		softmax.updateInLabels(batchLabel);
// 		softmax.fprop();
//  		softmax.bprop();
//  		softmax.update();
// 		printf("iter: %d current cost %.4f\n", it, softmax.getCurrentObjCost());
// 	}
// 	printf("training time %.2fs\n", ((double)cv::getTickCount() - exttime) / getTickFrequency());
// 
// 	Mat predictLabels;
// 	float accuracy;
// 
// 	batchData = Mat::zeros(digits.size(), 784, CV_8UC1);
// 	batchLabel = Mat::zeros(1, digits.size(), CV_8UC1);
// 	getBatchData(batchData, batchLabel, digits, labels, index, labels.cols, 0);
// 	opts::im2single(batchData, batchData);
// 	predict(predictLabels, accuracy, batchData, batchLabel, softmax.getWeights());
// 	printf("training accuracy %.2f%%\n", accuracy * 100);
// 
// 	imdb.loadDigits(testImageFile, digits);
// 	imdb.loadLabels(testLabelFile, labels);
// 	for (int i = 0; i < digits.size(); ++i) {
// 		index[i] = i;
// 	}
// 	batchData = Mat::zeros(digits.size(), 784, CV_8UC1);
// 	batchLabel = Mat::zeros(1, digits.size(), CV_8UC1);
// 	getBatchData(batchData, batchLabel, digits, labels, index, labels.cols, 0);
// 	opts::im2single(batchData, batchData);
// 	predict(predictLabels, accuracy, batchData, batchLabel, softmax.getWeights());
// 	printf("testing accuracy %.2f%%\n", accuracy * 100);

	return 0;
}