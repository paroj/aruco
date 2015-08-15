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
#include "marker.h"

#include <cstdio>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

namespace aruco {
/**
 *
 */
Marker::Marker() {
    id = -1;
    ssize = -1;
    for (int i = 0; i < 3; i++)
        Tvec(i) = Rvec(i) = -999999;
}

/**
 *
*/
Marker::Marker(const std::vector<cv::Point2f>& corners, int _id) : std::vector<cv::Point2f>(corners) {
    id = _id;
    ssize = -1;
    for (int i = 0; i < 3; i++)
        Tvec(i) = Rvec(i) = -999999;
}

void Marker::draw(Mat& in, Scalar color, int lineWidth, bool writeId) const {
    if (size() != 4)
        return;
    cv::line(in, (*this)[0], (*this)[1], color, lineWidth, CV_AA);
    cv::line(in, (*this)[1], (*this)[2], color, lineWidth, CV_AA);
    cv::line(in, (*this)[2], (*this)[3], color, lineWidth, CV_AA);
    cv::line(in, (*this)[3], (*this)[0], color, lineWidth, CV_AA);
    cv::rectangle(in, (*this)[0] - Point2f(2, 2), (*this)[0] + Point2f(2, 2), Scalar(0, 0, 255, 255),
                  lineWidth, CV_AA);
    cv::rectangle(in, (*this)[1] - Point2f(2, 2), (*this)[1] + Point2f(2, 2), Scalar(0, 255, 0, 255),
                  lineWidth, CV_AA);
    cv::rectangle(in, (*this)[2] - Point2f(2, 2), (*this)[2] + Point2f(2, 2), Scalar(255, 0, 0, 255),
                  lineWidth, CV_AA);
    if (writeId) {
        char cad[100];
        sprintf(cad, "id=%d", id);
        // determine the centroid
        Point cent(0, 0);
        for (int i = 0; i < 4; i++) {
            cent.x += (*this)[i].x;
            cent.y += (*this)[i].y;
        }
        cent.x /= 4.;
        cent.y /= 4.;
        putText(in, cad, cent, FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(255 - color[0], 255 - color[1], 255 - color[2], 255), 2);
    }
}

/**
 */
void Marker::calculateExtrinsics(float markerSize, const CameraParameters& CP, bool setYPerpendicular){
    CV_Assert( CP.isValid() && "invalid camera parameters. It is not possible to calculate extrinsics" );

    calculateExtrinsics(markerSize, CP.CameraMatrix, CP.Distorsion, setYPerpendicular);
}

void print(cv::Point3f p, string cad) { cout << cad << " " << p.x << " " << p.y << " " << p.z << endl; }
/**
 */
void Marker::calculateExtrinsics(float markerSizeMeters, cv::Mat camMatrix, cv::Mat distCoeff, bool setYPerpendicular) {

    CV_Assert( markerSizeMeters > 0 && isValid() && "invalid marker. It is not possible to calculate extrinsics");
    CV_Assert( camMatrix.rows != 0 || camMatrix.cols != 0 && "CameraMatrix is empty" );

    double halfSize = markerSizeMeters / 2.;
    cv::Matx<float, 4, 3> ObjPoints;
    ObjPoints(1, 0) = -halfSize;
    ObjPoints(1, 1) = halfSize;
    ObjPoints(1, 2) = 0;
    ObjPoints(2, 0) = halfSize;
    ObjPoints(2, 1) = halfSize;
    ObjPoints(2, 2) = 0;
    ObjPoints(3, 0) = halfSize;
    ObjPoints(3, 1) = -halfSize;
    ObjPoints(3, 2) = 0;
    ObjPoints(0, 0) = -halfSize;
    ObjPoints(0, 1) = -halfSize;
    ObjPoints(0, 2) = 0;

    cv::Matx<float, 4, 2> ImagePoints;

    // Set image points from the marker
    for (int c = 0; c < 4; c++) {
        ImagePoints(c, 0) = ((*this)[c].x);
        ImagePoints(c, 1) = ((*this)[c].y);
    }

    cv::Vec3d raux, taux;
    cv::solvePnP(ObjPoints, ImagePoints, camMatrix, distCoeff, raux, taux);
    Rvec = raux;
    Tvec = taux;
    // rotate the X axis so that Y is perpendicular to the marker plane
    if (setYPerpendicular)
        rotateXAxis(Rvec);
    ssize = markerSizeMeters;
    // cout<<(*this)<<endl;
}

/**
 */
cv::Point2f Marker::getCenter() const {
    cv::Point2f cent(0, 0);
    for (size_t i = 0; i < size(); i++) {
        cent.x += (*this)[i].x;
        cent.y += (*this)[i].y;
    }
    cent.x /= float(size());
    cent.y /= float(size());
    return cent;
}

/**
 */
float Marker::getArea() const {
    CV_Assert(size() == 4);
    // use the cross products
    cv::Point2f v01 = (*this)[1] - (*this)[0];
    cv::Point2f v03 = (*this)[3] - (*this)[0];
    float area1 = fabs(v01.x * v03.y - v01.y * v03.x);
    cv::Point2f v21 = (*this)[1] - (*this)[2];
    cv::Point2f v23 = (*this)[3] - (*this)[2];
    float area2 = fabs(v21.x * v23.y - v21.y * v23.x);
    return (area2 + area1) / 2.;
}
}
