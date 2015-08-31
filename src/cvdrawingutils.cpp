/*****************************
Copyright 2011 Rafael Muñoz Salinas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Rafael Muñoz Salinas.
********************************/
#include "cvdrawingutils.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

namespace aruco {
/****
 *
 *
 *
 ****/
void CvDrawingUtils::draw3dAxis(cv::Mat& Image, Marker& m, const CameraParameters& CP) {

    float size = m.ssize * 3;
    Mat_<float> objectPoints(4, 3);
    objectPoints(0, 0) = 0;
    objectPoints(0, 1) = 0;
    objectPoints(0, 2) = 0;
    objectPoints(1, 0) = size;
    objectPoints(1, 1) = 0;
    objectPoints(1, 2) = 0;
    objectPoints(2, 0) = 0;
    objectPoints(2, 1) = size;
    objectPoints(2, 2) = 0;
    objectPoints(3, 0) = 0;
    objectPoints(3, 1) = 0;
    objectPoints(3, 2) = size;

    vector<Point2f> imagePoints;
    cv::projectPoints(objectPoints, m.Rvec, m.Tvec, CP.CameraMatrix, CP.Distorsion, imagePoints);
    // draw lines of different colours
    cv::line(Image, imagePoints[0], imagePoints[1], Scalar(0, 0, 255, 255), 1, CV_AA);
    cv::line(Image, imagePoints[0], imagePoints[2], Scalar(0, 255, 0, 255), 1, CV_AA);
    cv::line(Image, imagePoints[0], imagePoints[3], Scalar(255, 0, 0, 255), 1, CV_AA);
    putText(Image, "x", imagePoints[1], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255, 255), 2);
    putText(Image, "y", imagePoints[2], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0, 255), 2);
    putText(Image, "z", imagePoints[3], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0, 255), 2);
}

/****
 *
 *
 *
 ****/
void CvDrawingUtils::draw3dCube(cv::Mat& Image, Marker& m, const CameraParameters& CP,
                                bool setYperpendicular) {
    Mat_<float> objectPoints(8, 3);
    double halfSize = m.ssize / 2;

    if (setYperpendicular) {
        objectPoints(0, 0) = -halfSize;
        objectPoints(0, 1) = 0;
        objectPoints(0, 2) = -halfSize;
        objectPoints(1, 0) = halfSize;
        objectPoints(1, 1) = 0;
        objectPoints(1, 2) = -halfSize;
        objectPoints(2, 0) = halfSize;
        objectPoints(2, 1) = 0;
        objectPoints(2, 2) = halfSize;
        objectPoints(3, 0) = -halfSize;
        objectPoints(3, 1) = 0;
        objectPoints(3, 2) = halfSize;

        objectPoints(4, 0) = -halfSize;
        objectPoints(4, 1) = m.ssize;
        objectPoints(4, 2) = -halfSize;
        objectPoints(5, 0) = halfSize;
        objectPoints(5, 1) = m.ssize;
        objectPoints(5, 2) = -halfSize;
        objectPoints(6, 0) = halfSize;
        objectPoints(6, 1) = m.ssize;
        objectPoints(6, 2) = halfSize;
        objectPoints(7, 0) = -halfSize;
        objectPoints(7, 1) = m.ssize;
        objectPoints(7, 2) = halfSize;
    } else {
        objectPoints(0, 0) = -halfSize;
        objectPoints(0, 1) = -halfSize;
        objectPoints(0, 2) = 0;
        objectPoints(1, 0) = halfSize;
        objectPoints(1, 1) = -halfSize;
        objectPoints(1, 2) = 0;
        objectPoints(2, 0) = halfSize;
        objectPoints(2, 1) = halfSize;
        objectPoints(2, 2) = 0;
        objectPoints(3, 0) = -halfSize;
        objectPoints(3, 1) = halfSize;
        objectPoints(3, 2) = 0;

        objectPoints(4, 0) = -halfSize;
        objectPoints(4, 1) = -halfSize;
        objectPoints(4, 2) = m.ssize;
        objectPoints(5, 0) = halfSize;
        objectPoints(5, 1) = -halfSize;
        objectPoints(5, 2) = m.ssize;
        objectPoints(6, 0) = halfSize;
        objectPoints(6, 1) = halfSize;
        objectPoints(6, 2) = m.ssize;
        objectPoints(7, 0) = -halfSize;
        objectPoints(7, 1) = halfSize;
        objectPoints(7, 2) = m.ssize;
    }

    vector<Point2f> imagePoints;
    projectPoints(objectPoints, m.Rvec, m.Tvec, CP.CameraMatrix, CP.Distorsion, imagePoints);
    // draw lines of different colours
    for (int i = 0; i < 4; i++)
        cv::line(Image, imagePoints[i], imagePoints[(i + 1) % 4], Scalar(0, 0, 255, 255), 1, CV_AA);

    for (int i = 0; i < 4; i++)
        cv::line(Image, imagePoints[i + 4], imagePoints[4 + (i + 1) % 4], Scalar(0, 0, 255, 255), 1, CV_AA);

    for (int i = 0; i < 4; i++)
        cv::line(Image, imagePoints[i], imagePoints[i + 4], Scalar(0, 0, 255, 255), 1, CV_AA);
}

/****
 *
 *
 *
 ****/
void CvDrawingUtils::draw3dAxis(cv::Mat& Image, Board& B, const CameraParameters& CP) {
    Mat_<float> objectPoints(4, 3);
    objectPoints(0, 0) = 0;
    objectPoints(0, 1) = 0;
    objectPoints(0, 2) = 0;
    objectPoints(1, 0) = 2 * B[0].ssize;
    objectPoints(1, 1) = 0;
    objectPoints(1, 2) = 0;
    objectPoints(2, 0) = 0;
    objectPoints(2, 1) = 2 * B[0].ssize;
    objectPoints(2, 2) = 0;
    objectPoints(3, 0) = 0;
    objectPoints(3, 1) = 0;
    objectPoints(3, 2) = 2 * B[0].ssize;

    vector<Point2f> imagePoints;
    projectPoints(objectPoints, B.Rvec, B.Tvec, CP.CameraMatrix, CP.Distorsion, imagePoints);
    // draw lines of different colours
    cv::line(Image, imagePoints[0], imagePoints[1], Scalar(0, 0, 255, 255), 2, CV_AA);
    cv::line(Image, imagePoints[0], imagePoints[2], Scalar(0, 255, 0, 255), 2, CV_AA);
    cv::line(Image, imagePoints[0], imagePoints[3], Scalar(255, 0, 0, 255), 2, CV_AA);

    putText(Image, "X", imagePoints[1], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255, 255), 2);
    putText(Image, "Y", imagePoints[2], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0, 255), 2);
    putText(Image, "Z", imagePoints[3], FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0, 255), 2);
}

/****
 *
 *
 *
 ****/
void CvDrawingUtils::draw3dCube(cv::Mat& Image, Board& B, const CameraParameters& CP,
                                bool setYperpendicular) {

    float cubeSize = B[0].ssize;
    float txz = -cubeSize / 2;
    Mat_<float> objectPoints(8, 3);

    if (setYperpendicular) {
        objectPoints(0, 0) = txz;
        objectPoints(0, 1) = 0;
        objectPoints(0, 2) = txz;
        objectPoints(1, 0) = txz + cubeSize;
        objectPoints(1, 1) = 0;
        objectPoints(1, 2) = txz;
        objectPoints(2, 0) = txz + cubeSize;
        objectPoints(2, 1) = cubeSize;
        objectPoints(2, 2) = txz;
        objectPoints(3, 0) = txz;
        objectPoints(3, 1) = cubeSize;
        objectPoints(3, 2) = txz;

        objectPoints(4, 0) = txz;
        objectPoints(4, 1) = 0;
        objectPoints(4, 2) = txz + cubeSize;
        objectPoints(5, 0) = txz + cubeSize;
        objectPoints(5, 1) = 0;
        objectPoints(5, 2) = txz + cubeSize;
        objectPoints(6, 0) = txz + cubeSize;
        objectPoints(6, 1) = cubeSize;
        objectPoints(6, 2) = txz + cubeSize;
        objectPoints(7, 0) = txz;
        objectPoints(7, 1) = cubeSize;
        objectPoints(7, 2) = txz + cubeSize;
    } else {
        objectPoints(0, 0) = txz;
        objectPoints(0, 2) = 0;
        objectPoints(0, 1) = txz;
        objectPoints(1, 0) = txz + cubeSize;
        objectPoints(1, 2) = 0;
        objectPoints(1, 1) = txz;
        objectPoints(2, 0) = txz + cubeSize;
        objectPoints(2, 2) = -cubeSize;
        objectPoints(2, 1) = txz;
        objectPoints(3, 0) = txz;
        objectPoints(3, 2) = -cubeSize;
        objectPoints(3, 1) = txz;

        objectPoints(4, 0) = txz;
        objectPoints(4, 2) = 0;
        objectPoints(4, 1) = txz + cubeSize;
        objectPoints(5, 0) = txz + cubeSize;
        objectPoints(5, 2) = 0;
        objectPoints(5, 1) = txz + cubeSize;
        objectPoints(6, 0) = txz + cubeSize;
        objectPoints(6, 2) = -cubeSize;
        objectPoints(6, 1) = txz + cubeSize;
        objectPoints(7, 0) = txz;
        objectPoints(7, 2) = -cubeSize;
        objectPoints(7, 1) = txz + cubeSize;
    }

    vector<Point2f> imagePoints;
    projectPoints(objectPoints, B.Rvec, B.Tvec, CP.CameraMatrix, CP.Distorsion, imagePoints);
    // draw lines of different colours
    for (int i = 0; i < 4; i++)
        cv::line(Image, imagePoints[i], imagePoints[(i + 1) % 4], Scalar(0, 0, 255, 255), 1, CV_AA);

    for (int i = 0; i < 4; i++)
        cv::line(Image, imagePoints[i + 4], imagePoints[4 + (i + 1) % 4], Scalar(0, 0, 255, 255), 1, CV_AA);

    for (int i = 0; i < 4; i++)
        cv::line(Image, imagePoints[i], imagePoints[i + 4], Scalar(0, 0, 255, 255), 1, CV_AA);
}
}
