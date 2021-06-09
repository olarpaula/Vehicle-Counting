#pragma once

#include "stdafx.h"
#include "common.h"
#include <random>
#include <math.h>


class Blob {
public:
	std::vector<Point> contour;
	Rect boundRect;
	Blob(std::vector<cv::Point> _contour);
	bool withMatch;
	bool isTracked;
	int frameCount;
	Point nextPos;
	void computeNextPos(void);
	std::vector<cv::Point> allPos;
	bool drawBoundRect;
};