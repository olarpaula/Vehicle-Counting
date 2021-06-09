#include "stdafx.h"
#include "Blob.h"

Blob::Blob(std::vector<cv::Point> _contour) {
	contour = _contour;
	boundRect = boundingRect(contour);
	isTracked = true;
	withMatch = true;
	frameCount = 0;
	Point center;
	center.x = (boundRect.x + boundRect.x + boundRect.width) / 2;
	center.y = (boundRect.y + boundRect.y + boundRect.height) / 2;
	allPos.push_back(center);
}

void Blob::computeNextPos() {
	int numPositions = (int)allPos.size();

	if (numPositions == 1) {
		nextPos.x = allPos.back().x;
		nextPos.y = allPos.back().y;

	}
	else {
		int limit = numPositions < 5 ? numPositions : 5;

		int sumX = 0, sumY = 0;
		int weight = 0;

		for (int i = 1; i < numPositions; i++) {
			sumX += (allPos[numPositions - i].x - allPos[numPositions - i - 1].x) * (numPositions - i);
			sumY += (allPos[numPositions - i].y - allPos[numPositions - i - 1].y) * (numPositions - i);
			weight += i;
		}

		int weightX = (int)std::round((float)sumX / (float)weight);
		int weightY = (int)std::round((float)sumY / (float)weight);

		nextPos.x = allPos.back().x + weightX;
		nextPos.y = allPos.back().y + weightY;

	}
}