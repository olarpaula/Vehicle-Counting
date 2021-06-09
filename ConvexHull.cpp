#include "stdafx.h"
#include "ConvexHull.h"

ConvexHull::ConvexHull(std::vector<cv::Point> _contour) {

	contour = _contour;

	boundingRect = cv::boundingRect(contour);

	centerPosition.x = (boundingRect.x + boundingRect.x + boundingRect.width) / 2;
	centerPosition.y = (boundingRect.y + boundingRect.y + boundingRect.height) / 2;



}