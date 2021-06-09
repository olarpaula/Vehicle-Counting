#pragma once

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

class ConvexHull {
public:
	std::vector<cv::Point> contour;

	cv::Rect boundingRect;

	cv::Point centerPosition;

	ConvexHull(std::vector<cv::Point> _contour);


};

