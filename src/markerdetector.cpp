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
#include "markerdetector.h"
#include "subpixelcorner.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include "arucofidmarkers.h"
#include "ar_omp.h"

using namespace std;
using namespace cv;

#define ARUCO_MARKER_BENCHMARK 0

namespace {
template <typename T>
void joinVectors(vector<vector<T> >& vv, vector<T>& v) {
    for (size_t i = 0; i < vv.size(); i++)
        v.insert(v.end(), vv[i].begin(), vv[i].end());
}

/**Given a vector vinout with elements and a boolean vector indicating the lements from it to remove,
 * this function remove the elements
 * @param vinout
 * @param toRemove
 */
template <typename T>
void removeElements(vector<T>& vinout, const vector<bool>& toRemove) {
    // remove the invalid ones by setting the valid in the positions left by the invalids
    size_t indexValid = 0;
    for (size_t i = 0; i < toRemove.size(); i++) {
        if (!toRemove[i]) {
            if (indexValid != i)
                vinout[indexValid] = vinout[i];
            indexValid++;
        }
    }
    vinout.resize(indexValid);
}

bool isInto(Mat& contour, vector<Point2f>& b) {

    for (size_t i = 0; i < b.size(); i++)
        if (pointPolygonTest(contour, b[i], false) > 0)
            return true;
    return false;
}

void findBestCornerInRegion_harris(cv::InputArray grey, vector<cv::Point2f>& Corners, int blockSize) {
    aruco::SubPixelCorner Subp;
    Subp.RefineCorner(grey, Corners);
}

// auxiliary functions to perform LINES refinement
void interpolate2Dline(const std::vector<Point2f>& inPoints, Point3f& outLine) {
    float minX, maxX, minY, maxY;
    minX = maxX = inPoints[0].x;
    minY = maxY = inPoints[0].y;
    for (unsigned int i = 1; i < inPoints.size(); i++) {
        if (inPoints[i].x < minX)
            minX = inPoints[i].x;
        if (inPoints[i].x > maxX)
            maxX = inPoints[i].x;
        if (inPoints[i].y < minY)
            minY = inPoints[i].y;
        if (inPoints[i].y > maxY)
            maxY = inPoints[i].y;
    }

    // create matrices of equation system
    Mat A(inPoints.size(), 2, CV_32FC1, Scalar(0));
    Mat B(inPoints.size(), 1, CV_32FC1, Scalar(0));
    Mat X;

    if (maxX - minX > maxY - minY) {
        // Ax + C = y
        for (int i = 0; i < inPoints.size(); i++) {

            A.at<float>(i, 0) = inPoints[i].x;
            A.at<float>(i, 1) = 1.;
            B.at<float>(i, 0) = inPoints[i].y;
        }

        // solve system
        solve(A, B, X, DECOMP_SVD);
        // return Ax + By + C
        outLine = Point3f(X.at<float>(0, 0), -1., X.at<float>(1, 0));
    } else {
        // By + C = x
        for (int i = 0; i < inPoints.size(); i++) {

            A.at<float>(i, 0) = inPoints[i].y;
            A.at<float>(i, 1) = 1.;
            B.at<float>(i, 0) = inPoints[i].x;
        }

        // solve system
        solve(A, B, X, DECOMP_SVD);
        // return Ax + By + C
        outLine = Point3f(-1., X.at<float>(0, 0), X.at<float>(1, 0));
    }
}

Point2f getCrossPoint(const cv::Point3f& line1, const cv::Point3f& line2) {
    // create matrices of equation system
    Matx22f A(line1.x, line1.y,
              line2.x, line2.y);
    Vec2f   B(-line1.z, -line2.z);

    return A.solve(B, DECOMP_SVD);
}

void distortPoints(vector<cv::Point2f> in, vector<cv::Point2f>& out, const Mat& camMatrix, const Mat& distCoeff) {
    // trivial extrinsics
    cv::Mat Rvec = cv::Mat(3, 1, CV_32FC1, cv::Scalar::all(0));
    cv::Mat Tvec = Rvec.clone();
    // calculate 3d points and then reproject, so opencv makes the distortion internally
    vector<cv::Point3f> cornersPoints3d;
    for (unsigned int i = 0; i < in.size(); i++)
        cornersPoints3d.push_back(
            cv::Point3f((in[i].x - camMatrix.at<float>(0, 2)) / camMatrix.at<float>(0, 0), // x
                        (in[i].y - camMatrix.at<float>(1, 2)) / camMatrix.at<float>(1, 1), // y
                        1));                                                               // z
    cv::projectPoints(cornersPoints3d, Rvec, Tvec, camMatrix, distCoeff, out);
}

// method to refine corner detection in case the internal border after threshold is found
// This was tested in the context of chessboard methods
void findCornerMaxima(vector<cv::Point2f>& Corners, const Mat& grey, int wsize) {
// for each element, search in a region around
#pragma omp parallel for
    for (int i = 0; i < int(Corners.size()); i++) {
        cv::Point2f minLimit(std::max(0, int(Corners[i].x - wsize)),
                             std::max(0, int(Corners[i].y - wsize)));
        cv::Point2f maxLimit(std::min(grey.cols, int(Corners[i].x + wsize)),
                             std::min(grey.rows, int(Corners[i].y + wsize)));

        cv::Mat reg = grey(cv::Range(minLimit.y, maxLimit.y), cv::Range(minLimit.x, maxLimit.x));
        cv::Mat harr, harrint;
        cv::cornerHarris(reg, harr, 3, 3, 0.04);

        // now, do a sum block operation
        cv::integral(harr, harrint);
        int bls_a = 4;
        for (int y = bls_a; y < harr.rows - bls_a; y++) {
            float* h = harr.ptr<float>(y);
            for (int x = bls_a; x < harr.cols - bls_a; x++)
                h[x] = harrint.at<double>(y + bls_a, x + bls_a) - harrint.at<double>(y + bls_a, x) -
                       harrint.at<double>(y, x + bls_a) + harrint.at<double>(y, x);
        }

        cv::Point2f best(-1, -1);
        cv::Point2f center(reg.cols / 2, reg.rows / 2);

        double maxv = 0;
        for (int i = 0; i < harr.rows; i++) {
            // L1 dist to center
            float* har = harr.ptr<float>(i);
            for (int x = 0; x < harr.cols; x++) {
                float d =
                    float(fabs(center.x - x) + fabs(center.y - i)) / float(reg.cols / 2 + reg.rows / 2);
                float w = 1. - d;
                if (w * har[x] > maxv) {
                    maxv = w * har[x];
                    best = cv::Point2f(x, i);
                }
            }
        }
        Corners[i] = best + minLimit;
    }
}

template<typename T>
void setPointIntoImage(cv::Point_<T>& p, cv::Size s) {
    if (p.x < 0)
        p.x = 0;
    else if (p.x >= s.width)
        p.x = s.width - 1;
    if (p.y < 0)
        p.y = 0;
    else if (p.y >= s.height)
        p.y = s.height - 1;
}

#if ARUCO_MARKER_DEBUG_DRAW
void drawContour(Mat& in, vector<Point>& contour, Scalar color) {
    for (size_t i = 0; i < contour.size(); i++) {
        cv::rectangle(in, contour[i], contour[i], color);
    }
}

void drawApproxCurve(Mat& in, vector<Point>& contour, Scalar color) {
    for (size_t i = 0; i < contour.size(); i++) {
        cv::line(in, contour[i], contour[(i + 1) % contour.size()], color);
    }
}
#endif
}

namespace aruco {
/************************************
 *
 *
 *
 *
 ************************************/
MarkerDetector::MarkerDetector() {
    _thresMethod = ADPT_THRES;
    _thresParam1 = _thresParam2 = 7;
    _cornerMethod = LINES;
    _useLockedCorners = false;
    //         _cornerMethod=SUBPIX;
    _thresParam1_range = 0;
    _markerWarpSize = 56;
    _speed = 0;
    markerIdDetectorFunc = aruco::FiducidalMarkers::detect;
    _minSize = 0.04;
    _maxSize = 0.5;

    _borderDistThres = 0.025; // corners in a border of 2.5% of image  are ignored
}
/************************************
 *
 *
 *
 *
 ************************************/

MarkerDetector::~MarkerDetector() {}

/************************************
 *
 *
 *
 *
 ************************************/
void MarkerDetector::setDesiredSpeed(int val) {
    if (val < 0)
        val = 0;
    else if (val > 3)
        val = 2;

    _speed = val;
    switch (_speed) {

    case 0:
        _markerWarpSize = 56;
        _cornerMethod = SUBPIX;
        break;

    case 1:
    case 2:
        _markerWarpSize = 28;
        _cornerMethod = NONE;
        break;
    };
}

/***
 *
 *
 **/
void MarkerDetector::enableLockedCornersMethod(bool enable) {
    _useLockedCorners = enable;
    if (enable)
        _cornerMethod = SUBPIX;
}
/************************************
 *
 * Main detection function. Performs all steps
 *
 *
 ************************************/
void MarkerDetector::detect(cv::InputArray input, vector<Marker>& detectedMarkers, Mat camMatrix,
                            Mat distCoeff, float markerSizeMeters, bool setYPerpendicular) {
    Mat grey;

    // it must be a 3 channel image
    if (input.type() == CV_8UC3)
        cv::cvtColor(input, grey, CV_BGR2GRAY);
    else
        grey = input.getMat();

#if ARUCO_MARKER_BENCHMARK
    double t1 = cv::getTickCount();
#endif
    //     cv::cvtColor(grey,_ssImC ,CV_GRAY2BGR); //DELETE

    Mat imgToBeThresHolded = grey;
    double ThresParam1 = _thresParam1, ThresParam2 = _thresParam2;

    /// Do threshold the image and detect contours
    // work simultaneouly in a range of values of the first threshold
    int n_param1 = 2 * _thresParam1_range + 1;
    vector<cv::Mat> thres_images(n_param1);

    if(n_param1 == 1) {
        thresHold(_thresMethod, imgToBeThresHolded, thres_images[0], ThresParam1, ThresParam2);
    } else {
#pragma omp parallel for
        for (int i = 0; i < n_param1; i++) {
            double t1 = ThresParam1 - _thresParam1_range + _thresParam1_range * i;
            thresHold(_thresMethod, imgToBeThresHolded, thres_images[i], t1, ThresParam2);
        }
    }
    thres = thres_images[n_param1 / 2];
    //

#if ARUCO_MARKER_BENCHMARK
    double t2 = cv::getTickCount();
#endif
    // find all rectangles in the thresholdes image
    vector<MarkerCandidate> MarkerCanditates;
    detectRectangles(thres_images, MarkerCanditates);

#if ARUCO_MARKER_BENCHMARK
    double t3 = cv::getTickCount();
#endif

    /// identify the markers
    //#pragma omp parallel for
    for (int i = 0; i < int(MarkerCanditates.size()); i++) {
        // Find proyective homography
        Mat canonicalMarker;
        warp(grey, canonicalMarker, Size(_markerWarpSize, _markerWarpSize), MarkerCanditates[i]);
        int nRotations;
        int id = markerIdDetectorFunc(canonicalMarker, nRotations);
        MarkerCanditates[i].id = id;

        if (id != -1) {
            if (_cornerMethod == LINES) // make LINES refinement before lose contour points
                refineCandidateLines(MarkerCanditates[i], camMatrix, distCoeff);

            // sort the points so that they are always in the same order no matter the camera
            // orientation
            std::rotate(MarkerCanditates[i].begin(),
                    MarkerCanditates[i].begin() + 4 - nRotations,
                    MarkerCanditates[i].end());
        }
    }

    // clear input data
    detectedMarkers.clear();
    _candidates.clear();

    // collect results based on id
    for (size_t i = 0; i < MarkerCanditates.size(); i++) {
        if(MarkerCanditates[i].id != -1) {
            detectedMarkers.push_back(MarkerCanditates[i]);
            detectedMarkers.back().id = MarkerCanditates[i].id;
        } else {
            _candidates.push_back(MarkerCanditates[i]);
        }
    }

#if ARUCO_MARKER_BENCHMARK
    double t4 = cv::getTickCount();
#endif
    /// refine the corner location if desired
    if (detectedMarkers.size() > 0 && _cornerMethod != NONE && _cornerMethod != LINES) {

        vector<Point2f> Corners;
        for (unsigned int i = 0; i < detectedMarkers.size(); i++)
            for (int c = 0; c < 4; c++)
                Corners.push_back(detectedMarkers[i][c]);

        // in case of "locked corners", it is neccesary in some ocasions to
        // find the corner in the sourrondings of the initially estimated location
        if (_useLockedCorners)
            findCornerMaxima(Corners, grey, _thresParam1);

        if (_cornerMethod == HARRIS)
            findBestCornerInRegion_harris(grey, Corners, 7);
        else if (_cornerMethod == SUBPIX) {
            cornerSubPix(grey, Corners, Size(_thresParam1, _thresParam1), Size(-1, -1),
                         TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS , 8, 0.005));
        }
        // copy back
        for (unsigned int i = 0; i < detectedMarkers.size(); i++)
            for (int c = 0; c < 4; c++)
                detectedMarkers[i][c] = Corners[i * 4 + c];
    }

#if ARUCO_MARKER_BENCHMARK
    double t5 = cv::getTickCount();
#endif

    // sort by id
    std::sort(detectedMarkers.begin(), detectedMarkers.end());
    // there might be still the case that a marker is detected twice because of the double border indicated
    // earlier,
    // detect and remove these cases
    vector<bool> toRemove(detectedMarkers.size(), false);
    for (int i = 0; i < int(detectedMarkers.size()) - 1; i++) {
        if (detectedMarkers[i].id == detectedMarkers[i + 1].id && !toRemove[i + 1]) {
            // deletes the one with smaller perimeter
            if (perimeter(detectedMarkers[i]) > perimeter(detectedMarkers[i + 1]))
                toRemove[i + 1] = true;
            else
                toRemove[i] = true;
        }
    }

    // remove markers with corners too near the image limits
    Point inputSize = Point(input.size());
    Rect validRegion(inputSize*_borderDistThres, inputSize*(1.0f - _borderDistThres));

    for (size_t i = 0; i < detectedMarkers.size(); i++) {
        // delete if any of the corners is too near image border
        for (size_t c = 0; c < detectedMarkers[i].size(); c++) {
            if(!validRegion.contains(detectedMarkers[i][c])) {
                toRemove[i] = true;
                break;
            }
        }
    }

    // remove the markers marker
    removeElements(detectedMarkers, toRemove);

    /// detect the position of detected markers if desired
    if (camMatrix.rows != 0 && markerSizeMeters > 0) {
#pragma omp parallel for
        for (unsigned int i = 0; i < detectedMarkers.size(); i++)
            detectedMarkers[i].calculateExtrinsics(markerSizeMeters, camMatrix, distCoeff,
                                                   setYPerpendicular);
    }

#if ARUCO_MARKER_BENCHMARK
    double t6 = cv::getTickCount();

    cerr << "Threshold: " << (t2 - t1) / double(cv::getTickFrequency()) << endl;
    cerr << "Rectangles: " << (t3 - t2) / double(cv::getTickFrequency()) << endl;
    cerr << "Identify: " << (t4 - t3) / double(cv::getTickFrequency()) << endl;
    cerr << "Subpixel: " << (t5 - t4) / double(cv::getTickFrequency()) << endl;
    cerr << "Filtering: " << (t6 - t5) / double(cv::getTickFrequency()) << endl;
#endif
}

/************************************
 *
 * Crucial step. Detects the rectangular regions of the thresholded image
 *
 *
 ************************************/
void MarkerDetector::detectRectangles(InputArray thres, vector<std::vector<cv::Point2f> >& MarkerCanditates) {
    vector<MarkerCandidate> candidates;
    vector<cv::Mat> thres_v(1, thres.getMat());
    detectRectangles(thres_v, candidates);
    // create the output
    MarkerCanditates.resize(candidates.size());
    for (size_t i = 0; i < MarkerCanditates.size(); i++)
        MarkerCanditates[i] = candidates[i];
}

void MarkerDetector::detectRectangles(vector<cv::Mat>& thresImgv, vector<MarkerCandidate>& OutMarkerCanditates) {
    //         omp_set_num_threads ( 1 );
    vector<vector<MarkerCandidate> > MarkerCanditatesV(thresImgv.size());
    // calcualte the min_max contour sizes
    int minSize = _minSize * std::max(thresImgv[0].cols, thresImgv[0].rows) * 4;
    int maxSize = _maxSize * std::max(thresImgv[0].cols, thresImgv[0].rows) * 4;

//         cv::Mat input;
//         cv::cvtColor ( thresImgv[0],input,CV_GRAY2BGR );

#pragma omp parallel for
    for (int t = 0; t < int(thresImgv.size()); t++) {
        std::vector<std::vector<cv::Point> > contours2;
        cv::Mat thres2;
        thresImgv[t].copyTo(thres2);
        cv::findContours(thres2, contours2, RETR_LIST, CHAIN_APPROX_NONE);

        vector<Point> approxCurve;
        /// for each contour, analyze if it is a paralelepiped likely to be the marker
        for (unsigned int i = 0; i < contours2.size(); i++) {
            // check it is a possible element by first checking is has enough points
            if (contours2[i].size() <= minSize || contours2[i].size() >= maxSize) {
                continue;
            }

            // approximate to a poligon
            approxPolyDP(contours2[i], approxCurve, double(contours2[i].size()) * 0.05, true);
            // 				drawApproxCurve(copy,approxCurve,Scalar(0,0,255));

            // check that the polygon has 4 points
            if (approxCurve.size() != 4) {
                continue;
            }
            /*
                                    drawContour ( input,contours2[i],Scalar ( 255,0,225 ) );
                                    namedWindow ( "input" );
                                    imshow ( "input",input );*/
            //  	 	waitKey(0);
            // and is convex
            if (!isContourConvex(approxCurve)) {
                continue;
            }

            // 					      drawApproxCurve(input,approxCurve,Scalar(255,0,255));
            // 						//ensure that the   distace between consecutive
            // points is large enough
            float minDist = 1e10;
            for (int j = 0; j < 4; j++) {
                float d = norm(approxCurve[i] - approxCurve[(i+1)%4]);
                if (d < minDist)
                    minDist = d;
            }

            // check that distance is not very small
            if (minDist <= 10) {
                continue;
            }

            // add the points
            // 	      cout<<"ADDED"<<endl;
            MarkerCanditatesV[t].push_back(Marker(vector<Point2f>(approxCurve.begin(), approxCurve.end())));
            MarkerCanditatesV[t].back().idx = i;
            MarkerCanditatesV[t].back().contour = contours2[i];
        }
    }
    // join all candidates
    vector<MarkerCandidate> MarkerCanditates;
    joinVectors(MarkerCanditatesV, MarkerCanditates);

    /// sort the points in anti-clockwise order
    vector<bool> swapped(MarkerCanditates.size(), false); // used later
    for (unsigned int i = 0; i < MarkerCanditates.size(); i++) {

        // trace a line between the first and second point.
        // if the thrid point is at the right side, then the points are anti-clockwise
        Point2f d1 = MarkerCanditates[i][1] - MarkerCanditates[i][0];
        Point2f d2 = MarkerCanditates[i][2] - MarkerCanditates[i][0];
        float o = (d1.x * d2.y) - (d1.y * d2.x);

        if (o < 0.0) { // if the third point is in the left side, then sort in anti-clockwise order
            swap(MarkerCanditates[i][1], MarkerCanditates[i][3]);
            swapped[i] = true;
            // sort the contour points
            //  	    reverse(MarkerCanditates[i].contour.begin(),MarkerCanditates[i].contour.end());//????
        }
    }

    /// remove these elements which corners are too close to each other
    // first detect candidates to be removed

    vector<vector<pair<int, int> > > TooNearCandidates_omp(omp_get_max_threads());
#pragma omp parallel for
    for (int i = 0; i < int(MarkerCanditates.size()); i++) {
        // 	cout<<"Marker i="<<i<<MarkerCanditates[i]<<endl;
        // calculate the average distance of each corner to the nearest corner of the other marker candidate
        for (unsigned int j = i + 1; j < MarkerCanditates.size(); j++) {
            float vdist[4];
            for (int c = 0; c < 4; c++)
                vdist[c] = norm(MarkerCanditates[i][c] - MarkerCanditates[j][c]);
            //                 dist/=4;
            // if distance is too small
            if (vdist[0] < 6 && vdist[1] < 6 && vdist[2] < 6 && vdist[3] < 6) {
                TooNearCandidates_omp[omp_get_thread_num()].push_back(pair<int, int>(i, j));
            }
        }
    }
    // join
    vector<pair<int, int> > TooNearCandidates;
    joinVectors(TooNearCandidates_omp, TooNearCandidates);
    // mark for removal the element of  the pair with smaller perimeter
    vector<bool> toRemove(MarkerCanditates.size(), false);
    for (unsigned int i = 0; i < TooNearCandidates.size(); i++) {
        if (perimeter(MarkerCanditates[TooNearCandidates[i].first]) >
            perimeter(MarkerCanditates[TooNearCandidates[i].second]))
            toRemove[TooNearCandidates[i].second] = true;
        else
            toRemove[TooNearCandidates[i].first] = true;
    }

    // remove the invalid ones
    // finally, assign to the remaining candidates the contour
    OutMarkerCanditates.reserve(MarkerCanditates.size());
    for (size_t i = 0; i < MarkerCanditates.size(); i++) {
        if (!toRemove[i]) {
            OutMarkerCanditates.push_back(MarkerCanditates[i]);
            //                 OutMarkerCanditates.back().contour=contours2[ MarkerCanditates[i].idx];
            if (swapped[i]) // if the corners where swapped, it is required to reverse here the points so
                            // that they are in the same order
                reverse(OutMarkerCanditates.back().contour.begin(),
                        OutMarkerCanditates.back().contour.end()); //????
        }
    }
    /*
            for ( size_t i=0; i<OutMarkerCanditates.size(); i++ )
                    OutMarkerCanditates[i].draw ( input,cv::Scalar ( 124,  255,125 ) );


            namedWindow ( "input" );
            imshow ( "input",input );*/
}

/************************************
 *
 *
 *
 *
 ************************************/
void MarkerDetector::thresHold(int method, InputArray grey, OutputArray out, double param1, double param2) {
    CV_Assert(grey.type() == CV_8UC1);

    if (param1 == -1)
        param1 = _thresParam1;
    if (param2 == -1)
        param2 = _thresParam2;

    switch (method) {
    case FIXED_THRES:
        cv::threshold(grey, out, param1, 255, THRESH_BINARY_INV);
        break;
    case ADPT_THRES: // currently, this is the best method
        // ensure that _thresParam1%2==1
        if (param1 < 3)
            param1 = 3;
        else if (((int)param1) % 2 != 1)
            param1 = (int)(param1 + 1);

        cv::adaptiveThreshold(grey, out, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, param1, param2);
        break;
    case CANNY: {
        // this should be the best method, and generally it is.
        // However, some times there are small holes in the marker contour that makes
        // the contour detector not to find it properly
        // if there is a missing pixel
        cv::Canny(grey, out, 10, 220);
        // I've tried a closing but it add many more points that some
        // times makes this even worse
        // 			  Mat aux;
        // 			  cv::morphologyEx(thres,aux,MORPH_CLOSE,Mat());
        // 			  out=aux;
    } break;
    }
}
/************************************
 *
 *
 *
 *
 ************************************/
void MarkerDetector::warp(InputArray in, OutputArray out, Size size, vector<Point2f> points){
    CV_Assert(points.size() == 4);

    Point2f pointsRes[] = {
        Point2f(0, 0),
        Point2f(size.width - 1, 0),
        Point2f(size.width - 1, size.height - 1),
        Point2f(0, size.height - 1)
    };

    // obtain the perspective transform
    Mat M = getPerspectiveTransform(points, Mat(4, 2, CV_32F, pointsRes));
    cv::warpPerspective(in, out, M, size, cv::INTER_NEAREST);
}

void findCornerPointsInContour(const vector<cv::Point2f>& points, const vector<cv::Point>& contour,
                               vector<int>& idxs) {
    CV_Assert(points.size() == 4);
    int idxSegments[4] = {-1, -1, -1, -1};
    // the first point coincides with one
    cv::Point points2i[4];
    for (int i = 0; i < 4; i++) {
        points2i[i].x = points[i].x;
        points2i[i].y = points[i].y;
    }

    for (size_t i = 0; i < contour.size(); i++) {
        if (idxSegments[0] == -1)
            if (contour[i] == points2i[0])
                idxSegments[0] = i;
        if (idxSegments[1] == -1)
            if (contour[i] == points2i[1])
                idxSegments[1] = i;
        if (idxSegments[2] == -1)
            if (contour[i] == points2i[2])
                idxSegments[2] = i;
        if (idxSegments[3] == -1)
            if (contour[i] == points2i[3])
                idxSegments[3] = i;
    }
    idxs.resize(4);
    for (int i = 0; i < 4; i++)
        idxs[i] = idxSegments[i];
}

int findDeformedSidesIdx(const vector<cv::Point>& contour, const vector<int>& idxSegments) {
    float distSum[4] = {0, 0, 0, 0};
    cv::Scalar colors[4] = {cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0),
                            cv::Scalar(111, 111, 0)};

    for (int i = 0; i < 3; i++) {
        cv::Point p1 = contour[idxSegments[i]];
        cv::Point p2 = contour[idxSegments[i + 1]];
        float inv_den = 1. / norm(p2 - p1);
        //   d=|v^^·r|=(|(x_2-x_1)(y_1-y_0)-(x_1-x_0)(y_2-y_1)|)/(sqrt((x_2-x_1)^2+(y_2-y_1)^2)).
        //         cerr<<"POSS="<<idxSegments[i]<<" "<<idxSegments[i+1]<<endl;
        for (size_t j = idxSegments[i]; j < idxSegments[i + 1]; j++) {
            float dist = std::fabs(float((p2.x - p1.x) * (p1.y - contour[j].y) -
                                         (p1.x - contour[j].x) * (p2.y - p1.y))) *
                         inv_den;
            distSum[i] += dist;
            //             cerr<< dist<<" ";
            //             cv::rectangle(_ssImC,contour[j],contour[j],colors[i],-1);
        }
        distSum[i] /= float(idxSegments[i + 1] - idxSegments[i]);
        //         cout<<endl<<endl;
    }

    // for the last one
    cv::Point p1 = contour[idxSegments[0]];
    cv::Point p2 = contour[idxSegments[3]];
    float inv_den = 1. / norm(p2 - p1);
    //   d=|v^^·r|=(|(x_2-x_1)(y_1-y_0)-(x_1-x_0)(y_2-y_1)|)/(sqrt((x_2-x_1)^2+(y_2-y_1)^2)).
    for (size_t j = 0; j < idxSegments[0]; j++)
        distSum[3] += std::fabs(float((p2.x - p1.x) * (p1.y - contour[j].y) -
                                      (p1.x - contour[j].x) * (p2.y - p1.y))) *
                      inv_den;
    for (size_t j = idxSegments[3]; j < contour.size(); j++)
        distSum[3] += std::fabs(float((p2.x - p1.x) * (p1.y - contour[j].y) -
                                      (p1.x - contour[j].x) * (p2.y - p1.y))) *
                      inv_den;

    distSum[3] /= float(idxSegments[0] + (contour.size() - idxSegments[3]));
    // now, get the maximum
    /*    for (int i=0;i<4;i++)
            cout<<"DD="<<distSum[i]<<endl;*/
    // check the two combinations to see the one with higher error
    if (distSum[0] + distSum[2] > distSum[1] + distSum[3])
        return 0;
    else
        return 1;
}

/************************************
 *
 *
 *
 *
 ************************************/
bool MarkerDetector::warp_cylinder(Mat& in, Mat& out, Size size, MarkerCandidate& mcand){

    CV_Assert(mcand.size() == 4);

    // check first the real need for cylinder warping
    //     cout<<"im="<<mcand.contour.size()<<endl;

    //     for (size_t i=0;i<mcand.contour.size();i++) {
    //         cv::rectangle(_ssImC ,mcand.contour[i],mcand.contour[i],cv::Scalar(111,111,111),-1 );
    //     }
    //     mcand.draw(imC,cv::Scalar(0,255,0));
    // find the 4 different segments of the contour
    vector<int> idxSegments;
    findCornerPointsInContour(mcand, mcand.contour, idxSegments);
    // let us rearrange the points so that the first corner is the one whith smaller idx
    int minIdx = 0;
    for (int i = 1; i < 4; i++)
        if (idxSegments[i] < idxSegments[minIdx])
            minIdx = i;
    // now, rotate the points to be in this order
    std::rotate(idxSegments.begin(), idxSegments.begin() + minIdx, idxSegments.end());
    std::rotate(mcand.begin(), mcand.begin() + minIdx, mcand.end());

    //     cout<<"idxSegments="<<idxSegments[0]<< " "<<idxSegments[1]<< " "<<idxSegments[2]<<"
    //     "<<idxSegments[3]<<endl;
    // now, determine the sides that are deformated by cylinder perspective
    int defrmdSide = findDeformedSidesIdx(mcand.contour, idxSegments);
    //     cout<<"Def="<<defrmdSide<<endl;

    // instead of removing perspective distortion  of the rectangular region
    // given by the rectangle, we enlarge it a bit to include the deformed parts
    cv::Point2f center = mcand.getCenter();
    Point2f enlargedRegion[4];
    for (int i = 0; i < 4; i++)
        enlargedRegion[i] = mcand[i];
    if (defrmdSide == 0) {
        enlargedRegion[0] = mcand[0] + (mcand[3] - mcand[0]) * 1.2;
        enlargedRegion[1] = mcand[1] + (mcand[2] - mcand[1]) * 1.2;
        enlargedRegion[2] = mcand[2] + (mcand[1] - mcand[2]) * 1.2;
        enlargedRegion[3] = mcand[3] + (mcand[0] - mcand[3]) * 1.2;
    } else {
        enlargedRegion[0] = mcand[0] + (mcand[1] - mcand[0]) * 1.2;
        enlargedRegion[1] = mcand[1] + (mcand[0] - mcand[1]) * 1.2;
        enlargedRegion[2] = mcand[2] + (mcand[3] - mcand[2]) * 1.2;
        enlargedRegion[3] = mcand[3] + (mcand[2] - mcand[3]) * 1.2;
    }
    for (size_t i = 0; i < 4; i++)
        setPointIntoImage(enlargedRegion[i], in.size());

    /*
        cv::Scalar
       colors[4]={cv::Scalar(0,0,255),cv::Scalar(255,0,0),cv::Scalar(0,255,0),cv::Scalar(111,111,0)};
        for (int i=0;i<4;i++) {
            cv::rectangle(_ssImC,mcand.contour[idxSegments[i]]-cv::Point(2,2),mcand.contour[idxSegments[i]]+cv::Point(2,2),colors[i],-1
       );
            cv::rectangle(_ssImC,enlargedRegion[i]-cv::Point2f(2,2),enlargedRegion[i]+cv::Point2f(2,2),colors[i],-1
       );

        }*/
    //     cv::imshow("imC",_ssImC);

    // calculate the max distance from each contour point the line of the corresponding segment it belongs
    // to
    //     calculate
    //      cv::waitKey(0);
    // check that the region is into image limits
    // obtain the perspective transform
    Point2f pointsRes[4], pointsIn[4];
    for (int i = 0; i < 4; i++)
        pointsIn[i] = mcand[i];

    cv::Size enlargedSize = size;
    enlargedSize.width += 2 * enlargedSize.width * 0.2;
    pointsRes[0] = (Point2f(0, 0));
    pointsRes[1] = Point2f(enlargedSize.width - 1, 0);
    pointsRes[2] = Point2f(enlargedSize.width - 1, enlargedSize.height - 1);
    pointsRes[3] = Point2f(0, enlargedSize.height - 1);
    // rotate to ensure that deformed sides are in the horizontal axis when warping
    if (defrmdSide == 0)
        rotate(pointsRes, pointsRes + 1, pointsRes + 4);
    cv::Mat imAux, imAux2(enlargedSize, CV_8UC1);
    Mat M = cv::getPerspectiveTransform(enlargedRegion, pointsRes);
    cv::warpPerspective(in, imAux, M, enlargedSize, cv::INTER_NEAREST);

    // now, transform all points to the new image
    vector<cv::Point> pointsCO(mcand.contour.size());
    assert(M.type() == CV_64F);
    assert(M.cols == 3 && M.rows == 3);
    //     cout<<M<<endl;
    double* mptr = M.ptr<double>(0);
    imAux2.setTo(cv::Scalar::all(0));

    for (size_t i = 0; i < mcand.contour.size(); i++) {
        float inX = mcand.contour[i].x;
        float inY = mcand.contour[i].y;
        float w = inX * mptr[6] + inY * mptr[7] + mptr[8];
        cv::Point2f pres;
        pointsCO[i].x = ((inX * mptr[0] + inY * mptr[1] + mptr[2]) / w) + 0.5;
        pointsCO[i].y = ((inX * mptr[3] + inY * mptr[4] + mptr[5]) / w) + 0.5;
        // make integers
        setPointIntoImage(pointsCO[i], imAux.size()); // ensure points are into image limits
        // 	cout<<"p="<<pointsCO[i]<<" "<<imAux.size().width<<" "<<imAux.size().height<<endl;
        imAux2.at<uchar>(pointsCO[i].y, pointsCO[i].x) = 255;
        if (pointsCO[i].y > 0)
            imAux2.at<uchar>(pointsCO[i].y - 1, pointsCO[i].x) = 255;
        if (pointsCO[i].y < imAux2.rows - 1)
            imAux2.at<uchar>(pointsCO[i].y + 1, pointsCO[i].x) = 255;
    }

    cv::Mat outIm(enlargedSize, CV_8UC1);
    outIm.setTo(cv::Scalar::all(0));
    // now, scan in lines to determine the required displacement
    for (int y = 0; y < imAux2.rows; y++) {
        uchar* _offInfo = imAux2.ptr<uchar>(y);
        int start = -1, end = -1;
        // determine the start and end of markerd regions
        for (int x = 0; x < imAux.cols; x++) {
            if (_offInfo[x]) {
                if (start == -1)
                    start = x;
                else
                    end = x;
            }
        }
        //       cout<<"S="<<start<<" "<<end<<" "<<end-start<<" "<<(size.width>>1)<<endl;
        // check that the size is big enough and
        assert(start != -1 && end != -1 && (end - start) > size.width >> 1);
        uchar* In_image = imAux.ptr<uchar>(y);
        uchar* Out_image = outIm.ptr<uchar>(y);
        memcpy(Out_image, In_image + start, imAux.cols - start);
    }

    //     cout<<"SS="<<mcand.contour.size()<<" "<<pointsCO.size()<<endl;
    // get the central region with the size specified
    cv::Mat centerReg = outIm(cv::Range::all(), cv::Range(0, size.width));
    out = centerReg.clone();
    //     cv::perspectiveTransform(mcand.contour,pointsCO,M);
    // draw them
    //     cv::imshow("out2",out);
    //     cv::imshow("imm",imAux2);
    //     cv::waitKey(0);
    return true;
}

/**
 *
 *
 */
void MarkerDetector::refineCandidateLines(MarkerDetector::MarkerCandidate& candidate,
                                          const cv::Mat& camMatrix, const cv::Mat& distCoeff) {
    // search corners on the contour vector
    int cornerIndex[4];
    for (size_t j = 0; j < candidate.contour.size(); j++) {
        for (int k = 0; k < 4; k++) {
            if (candidate.contour[j] == Point(candidate[k])) {
                cornerIndex[k] = j;
            }
        }
    }

    // contour pixel in inverse order or not?
    bool inverse;
    if ((cornerIndex[1] > cornerIndex[0]) &&
        (cornerIndex[2] > cornerIndex[1] || cornerIndex[2] < cornerIndex[0]))
        inverse = false;
    else if (cornerIndex[2] > cornerIndex[1] && cornerIndex[2] < cornerIndex[0])
        inverse = false;
    else
        inverse = true;

    // get pixel vector for each line of the marker
    int inc = inverse ? -1 : 1;

    // undistort contour
    vector<Point2f> contour2f(candidate.contour.begin(), candidate.contour.end());
    if (!camMatrix.empty() && !distCoeff.empty())
        cv::undistortPoints(contour2f, contour2f, camMatrix, distCoeff, cv::Mat(), camMatrix);

    vector<cv::Point2f> contourLines[4];
    for (int l = 0; l < 4; l++) {
        int j = cornerIndex[l];
        while (j != cornerIndex[(l + 1) % 4]) {
            contourLines[l].push_back(contour2f[j]);

            j = (j + inc) % contour2f.size();
            if (j < 0)
                j += contour2f.size();
        }

        // fix up line equation by adding the next corner
        // if there were no intermediate points
        if(contourLines[l].size() == 1) {
            contourLines[l].push_back(contour2f[cornerIndex[(l + 1) % 4]]);
        }
    }

    // interpolate marker lines
    Point3f lines[4];
    for (int j = 0; j < 4; j++)
        interpolate2Dline(contourLines[j], lines[j]);

    // get cross points of lines
    vector<Point2f> crossPoints(4);
    for (int i = 0; i < 4; i++)
        crossPoints[i] = getCrossPoint(lines[i], lines[((i - 1) % 4 + 4) % 4]);

    // distort corners again if undistortion was performed
    if (!camMatrix.empty() && !distCoeff.empty())
        distortPoints(crossPoints, crossPoints, camMatrix, distCoeff);

    // reassing points
    for (int j = 0; j < 4; j++) {
        candidate[j] = crossPoints[j];
    }
}

/* Attempt to make it faster than in opencv. I could not :( Maybe trying with SSE3...
void MarkerDetector::warpPerspective(const cv::Mat &in,cv::Mat & out, const cv::Mat & M,cv::Size size)
{
   //inverse the matrix
   out.create(size,in.type());
   //convert to float to speed up operations
   const double *m=M.ptr<double>(0);
   float mf[9];
   mf[0]=m[0];mf[1]=m[1];mf[2]=m[2];
   mf[3]=m[3];mf[4]=m[4];mf[5]=m[5];
   mf[6]=m[6];mf[7]=m[7];mf[8]=m[8];

   for(int y=0;y<out.rows;y++){
     uchar *_ptrout=out.ptr<uchar>(y);
     for(int x=0;x<out.cols;x++){
   //get the x,y position
   float den=1./(x*mf[6]+y*mf[7]+mf[8]);
   float ox= (x*mf[0]+y*mf[1]+mf[2])*den;
   float oy= (x*mf[3]+y*mf[4]+mf[5])*den;
   _ptrout[x]=in.at<uchar>(oy,ox);
     }
   }
}
*/

/************************************
*
*
*
*
************************************/

void MarkerDetector::setMinMaxSize(float min, float max) {
    CV_Assert (min > 0 && min <= 1);
    CV_Assert (max > 0 && max <= 1);
    CV_Assert (min < max);

    _minSize = min;
    _maxSize = max;
}

/************************************
*
*
*
*
************************************/

void MarkerDetector::setWarpSize(int val) {
    CV_Assert (val >= 10);

    _markerWarpSize = val;
}
};
