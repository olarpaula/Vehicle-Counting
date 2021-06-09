#include "stdafx.h"
#include "Blob.h"
#include<iostream>     
#include <random>
#include <math.h>
#include <stdlib.h>

void trackedCar(std::vector<Blob>& cars, int& limit, int& carCount, int& upCars, int& downCars, int& lineCrossed);
void computeConvolutionProd(Mat src, Mat core, Mat& dst);
void color2gray(Mat& colorSrc, Mat& dstGray);
void showCount(int& carCount, int& upCars, int& downCars, Mat& imgFrame2Copy);
void treshBinary(Mat& src, Mat& dst);
void computeFrameDiff(Mat& frame1, Mat& frame2, Mat& frameDiff);
void dilateMat(Mat& src, Mat& dst);
void erodeMat(Mat& src, Mat& dst);
void drawBlobs(Size image, std::vector<Blob> cars);
void drawBoundingRect(std::vector<Blob>& cars, Mat& imgFrame2Copy);
void computeMatForContours(Mat& src, Mat& dst);
void checkIfTracked(std::vector<Blob>& cars);
void compareFrames(std::vector<Blob>& allCars, std::vector<Blob>& newCars);
void reinitializeCars(std::vector<Blob>& cars);
void closeMat(Mat& src, Mat& dst);
Mat gaussianFilter(Mat& src);

int main(void) {
	
	VideoCapture video;

	//process two consecutive frames
	Mat frame1;
	Mat frame2;

	//car counting (total, up and down the road)
	int carCount = 0;
	int upCars = 0;
	int downCars = 0;

	std::vector<Blob> cars;
	Point lineEdges[2];

	//open video file and start processing
	video.open("video.mp4");

	if (!video.isOpened()) {
		std::cout << "err opening video file" << std::endl;
		return -1;
	}

	if (video.get(CAP_PROP_FRAME_COUNT) < 2) {
		std::cout << "video must have at least two frames";
		return -1;
	}

	//process two consecutive frames
	video.read(frame1);
	video.read(frame2);

	//limit Line y axis position
	int limit = 250;

	//end video
	char chCheckForEscKey = 0;

	bool firstFrame = true;


	while (video.isOpened()) {

		std::vector<Blob> newCars;

		resize(frame1, frame1, Size(600, 400));
		resize(frame2, frame2, Size(600, 400));

		Mat frame1copy, frame2copy;

		//transform frame from rgb to grayscale
		color2gray(frame1, frame1copy);
		color2gray(frame2, frame2copy);

		///imshow("Color2Gray", frame1copy);
		///imshow("Cars", frame1);

		//apply gaussian filter 3x3 or 5x5 for noise elimination from the image
		frame1copy = gaussianFilter(frame1copy);
		frame2copy = gaussianFilter(frame2copy);

		///imshow("GaussianFilter", frame1copy);

		//commpute frame difference to get the objects in motion. By substracting two consecutive frames, the fixed pixels are canceled
		//compute frame treshold to visualize the remaiing pixels after substracting the frames, aka the moving cars
		Mat frameDiff, frameTreshold;

		computeFrameDiff(frame1copy, frame2copy, frameDiff);

		treshBinary(frameDiff, frameTreshold);

		///imshow("frameDiff", frameDiff);
		///imshow("frameTreshold", frameTreshold);

		//the output of first operation: backgroung substraction
		//imshow("background substraction", frameTreshold);
		
		//compute contour of pixels for objects that are moving (using dilate and erode)
		computeMatForContours(frameTreshold, frameTreshold);
		
		//vector of points vector for cars contour  
		std::vector<std::vector<Point> > contours;
		findContours(frameTreshold, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	    
		//compute cars contours (just for visualizing)
		//Mat frameContour(frameTreshold.size(), CV_8UC3, Scalar(0.0, 0.0, 0.0));
		//drawContours(frameContour, contours, -1, Scalar(255.0, 255.0, 255.0), 1);
		//imshow("contur", frameContour);

		//compute the blobs of cars
		std::vector<std::vector<Point> > outputHulls(contours.size());
		for (unsigned int i = 0; i < contours.size(); i++) {
			convexHull(contours[i], outputHulls[i]);
		}

		//if a blob has the specifications for a car, add to list
		for (int i = 0; i < outputHulls.size(); i++) {
			Blob isBlob(outputHulls[i]);

			if (isBlob.boundRect.area() > 70 && isBlob.boundRect.height > 15 && isBlob.boundRect.height < 72 && isBlob.boundRect.width > 15 && isBlob.boundRect.width < 112) {
				newCars.push_back(isBlob);
			}
		}

		//draw blobs of the cars
		//drawBlobs(frameTreshold.size(), newCars);

		//in the first frame, store every cars information
		if (firstFrame == true) {
			for (unsigned i = 0; i < newCars.size(); i++) {
				cars.push_back(newCars[i]);
			}
		}
		//in the next frames, try to match every frame to the other one based on cars position and movement
		else {
			reinitializeCars(cars);
			compareFrames(cars, newCars);
		}

		//draw blobs of the cars
		//drawBlobs(frameTreshold.size(), cars);

		Mat carsFrame = frame2.clone();     

		//draw bounding boxes of the cars
		//drawBoundingRect(cars, frame2copy); 
		
		//if a car has crossed the line, color white
		int lineCrossed;
		trackedCar(cars, limit, carCount, upCars, downCars, lineCrossed);
		if (lineCrossed) {
			line(carsFrame, Point(0, limit), Point(frame1.cols-1, limit), Scalar(255.0, 255.0, 255.0), 1);
			lineCrossed = 0;
		}
		else {
			line(carsFrame, Point(0, limit), Point(frame1.cols-1, limit), Scalar(0.0, 0.0, 0.0), 1);
		}

		//show total sum of passing cars
		showCount(carCount, upCars, downCars, carsFrame);

		//final result
		imshow("CARS", carsFrame);

		//get rid of current frame tracked cars
		newCars.clear();
		
		//process the enxt frame
		frame1 = frame2.clone();  

		if ((video.get(CAP_PROP_POS_FRAMES) + 1) < video.get(CAP_PROP_FRAME_COUNT)) {
			video.read(frame2);
		}
		else {
			std::cout << "end of video" << std::endl;
			break;
		}

		firstFrame = false;

		chCheckForEscKey = waitKey(1);  

		if (chCheckForEscKey == 27)
			break;

	}

	if (chCheckForEscKey != 27) {
		waitKey(0);
	}

	return(0);
}

void computeConvolutionProd(Mat src, Mat core, Mat& dst) {
	dst = src.clone();
	int k = (core.cols - 1) / 2;

	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {

			int sum = 0;

			for (int m = 0; m < core.cols; m++) {
				for (int n = 0; n < core.cols; n++) {
					sum += core.at<float>(m, n) * src.at<uchar>(i + m - k, j + n - k);
				}
			}

			dst.at<uchar>(i, j) = sum;
		}
	}
	//return dst;
}

void color2gray(Mat& src, Mat& dst) {
	//Mat dstGray;
	dst = Mat(src.rows, src.cols, CV_8UC1);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			Vec3b currPixel = src.at<Vec3b>(i, j);
			dst.at<uchar>(i, j) = (currPixel[0] + currPixel[1] + currPixel[2]) / 3;
		}
	}

	//return dstGray;
}

void reinitializeCars(std::vector<Blob>& cars) {
	for (unsigned i = 0; i < cars.size(); i++) {
		cars[i].withMatch = false;
		cars[i].computeNextPos();
	}
}
void compareFrames(std::vector<Blob>& allCars, std::vector<Blob>& newCars) {

	for (unsigned i=0; i < newCars.size(); i++) {
		Blob car = newCars[i];
		
		int minPos = 0;
		double minDist = LONG_MAX;

		for (unsigned int i = 0; i < allCars.size(); i++) {
			if (allCars[i].isTracked == true) {

				int dx = abs(car.allPos.back().x - allCars[i].nextPos.x);
				int dy = abs(car.allPos.back().y - allCars[i].nextPos.y);

				double pointsDistance = sqrt(pow(dx,2) + pow(dy,2));

				if (pointsDistance < minDist) {
					minDist = pointsDistance;
					minPos = i;
				}
			}
		}

		if (minDist < car.boundRect.height * 1.15) {
			allCars[minPos].allPos.push_back(car.allPos.back());
			allCars[minPos].withMatch = true;
		}
		else { //add new blob		
			car.withMatch = true;
			allCars.push_back(car);
		}
	}

	checkIfTracked(allCars);
}

void drawBlobs(Size imageSize, std::vector<Blob> blobs) {

	Mat image(imageSize, CV_8UC3, Scalar(0.0, 0.0, 0.0));

	std::vector<std::vector<Point> > contours;

	for(unsigned i=0; i < blobs.size(); i++) {
		contours.push_back(blobs[i].contour);	
	}

	drawContours(image, contours, -1, Scalar(255.0, 255.0, 255.0), -1);

	imshow("Blobs", image);
}


void trackedCar(std::vector<Blob>& cars, int& limit, int& carCount, int& upCars, int& downCars, int& lineCrossed) {

	for (unsigned i = 0; i < cars.size(); i++) {
		Blob blob = cars[i];

		if (blob.isTracked == true && blob.allPos.size() >= 2) {
			int lastPos = (int)blob.allPos.size() - 2;
			int b4lastPos = (int)blob.allPos.size() - 1;

			if (blob.allPos[lastPos].y > limit&& blob.allPos[b4lastPos].y <= limit) {
				carCount++;
				upCars++;
				//cars.pop_back();
				blob.isTracked = false;
				lineCrossed = 1;
			}

			else if (blob.allPos[lastPos].y < limit&& blob.allPos[b4lastPos].y >= limit) {
				carCount++;
				downCars++;
				//cars.pop_back();
				blob.isTracked = false;
				lineCrossed = 1;
			}
		}
	}
}

void showCount(int& carCount, int& upCars, int& downCars, Mat& imgFrame2Copy) {

	int fontStyle = CV_FONT_HERSHEY_SIMPLEX;
	double fontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 400000.0;
	int fontThickness = (int)std::round(fontScale * 1.5);

	Size textSize = getTextSize(std::to_string(carCount), fontStyle, fontScale, fontThickness, 0);

	Point carsPos, upCarsPos, downCarsPos;

	carsPos.x = imgFrame2Copy.cols - 1 - (int)((double)textSize.width * 1.25) - 60;
	carsPos.y = (int)((double)textSize.height * 1.25);

	upCarsPos.x = carsPos.x;
	upCarsPos.y = carsPos.y + 20;

	downCarsPos.x = carsPos.x;
	downCarsPos.y = carsPos.y + 40;

	putText(imgFrame2Copy, "Total " + std::to_string(carCount), carsPos, fontStyle, fontScale, Scalar(0.0, 0.0, 0.0), fontThickness);
	putText(imgFrame2Copy, "Up   " + std::to_string(upCars), upCarsPos, fontStyle, fontScale, Scalar(0.0, 0.0, 0.0), fontThickness);
	putText(imgFrame2Copy, "Down " + std::to_string(downCars), downCarsPos, fontStyle, fontScale, Scalar(0.0, 0.0, 0.0), fontThickness);
}

void treshBinary(Mat& src, Mat& dst) {
	dst = src.clone();

	//int treshold = 30; //pt cars
	int treshold = 18; //pt video

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			if (dst.at<uchar>(i, j) < treshold) {
				dst.at<uchar>(i, j) = 0;
			}
			else {
				dst.at<uchar>(i, j) = 255;
			}
		}
	}

	//return dst;
}

void computeFrameDiff(Mat& frame1, Mat& frame2, Mat& frameDiff) {
	frameDiff = Mat(frame1.rows, frame1.cols, CV_8UC1);

	for (int i = 0; i < frame1.rows; i++) {
		for (int j = 0; j < frame1.cols; j++) {

			frameDiff.at<uchar>(i, j) = abs(frame1.at<uchar>(i, j) - frame2.at<uchar>(i, j));
		}
	}

	//return frameDiff;
}

void dilateMat(Mat& src, Mat& dst) {
	dst = src.clone();

	int di[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };
	int dj[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };

	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {

			if (src.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 8; k++) {
					dst.at<uchar>(i + di[k], j + dj[k]) = 0;
				}
			}
		}
	}

	//return dst;
}

void erodeMat(Mat& src, Mat& dst) {
	dst = src.clone();

	int di[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };
	int dj[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };

	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {

			if (src.at<uchar>(i, j) == 0) {
				int countNeighb = 0;

				for (int k = 0; k < 8; k++) {
					if (src.at<uchar>(i + di[k], j + dj[k]) == 0) {
						countNeighb++;
					}
				}
				if (countNeighb == 8) {
					dst.at<uchar>(i, j) = 0;
				}
				else {
					dst.at<uchar>(i, j) = 255;
				}
			}
		}
	}

	//return dst;
}

void drawBoundingRect(std::vector<Blob>& cars, Mat& imgFrame2Copy) {

	for (unsigned int i = 0; i < cars.size(); i++) {

		if (cars[i].isTracked == true) {
			rectangle(imgFrame2Copy, cars[i].boundRect, Scalar(0.0, 0.0, 0.0), 1);
		}
	}
}

Mat gaussianFilter(Mat& src) {
	Mat dst;

	float gaus[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
	//float gaus[25] = { 1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1 };

	Mat_<float> nucleu(3, 3, gaus);
	//Mat_<float> nucleu(5, 5, gaus);

	for (int i = 0; i < nucleu.rows; i++) {
		for (int j = 0; j < nucleu.cols; j++) {

			nucleu.at<float>(i, j) /= 16.0;
			//nucleu.at<float>(i, j) /= 273.0;
		}
	}

	computeConvolutionProd(src, nucleu, dst);

	return dst;
}

void checkIfTracked(std::vector<Blob>& cars) {
	for (unsigned i = 0; i < cars.size(); i++) {

		if (cars[i].withMatch == false) {
			cars[i].frameCount++;
		}

		if (cars[i].frameCount >= 5) {
			cars[i].isTracked = false;
		}

	}
}

void computeMatForContours(Mat& src, Mat& dst) {
	Mat structuringElement5x5 = getStructuringElement(MORPH_RECT, Size(5, 5));

	dilate(src, dst, structuringElement5x5);
	dilate(dst, dst, structuringElement5x5);
	erode(dst, dst, structuringElement5x5);
	dilate(dst, dst, structuringElement5x5);
	dilate(dst, dst, structuringElement5x5);
	erode(dst, dst, structuringElement5x5);

	/*src = dilateMat(src);
	src = closeMat(src);
	src = dilateMat(src);
	src = closeMat(src);*/

	//return src;
}

void closeMat(Mat& src, Mat& dst) {

	Mat dil;
	dilateMat(src, dil);
	erodeMat(dil, dst);

	//return dst;
}
