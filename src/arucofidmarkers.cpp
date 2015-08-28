/**

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

*/
#include "arucofidmarkers.h"
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;

namespace {
typedef Matx<bool, 5, 5> MarkerCode;

vector<int> getListOfValidMarkersIds_random(size_t nMarkers, const vector<int>& excluded) {
    CV_Assert(nMarkers + excluded.size() <= 1024 && "Number of possible markers is exceeded");

    int listOfMarkers[1024];
    // set a list with all ids
    for (int i = 0; i < 1024; i++)
        listOfMarkers[i] = i;

    for (size_t i = 0; i < excluded.size(); i++)
        listOfMarkers[excluded[i]] = -1;
    // random shuffle
    random_shuffle(listOfMarkers, listOfMarkers + 1024, theRNG());
    // now, take the first  nMarkers elements with value !=-1
    int i = 0;
    vector<int> retList;
    while (retList.size() < nMarkers) {
        if (listOfMarkers[i] != -1)
            retList.push_back(listOfMarkers[i]);
        i++;
    }
    return retList;
}

MarkerCode rotate(const MarkerCode& in) {
    MarkerCode out = in;

    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            out(i, j) = in(in.cols - j - 1, i);
        }
    }
    return out;
}

int hammDistMarker(const MarkerCode& bits) {
    bool ids[4][5] = {{1, 0, 0, 0, 0}, {1, 0, 1, 1, 1}, {0, 1, 0, 0, 1}, {0, 1, 1, 1, 0}};
    int dist = 0;

    for (int y = 0; y < 5; y++) {
        int minSum = 1e5;
        // hamming distance to each possible word
        for (int p = 0; p < 4; p++) {
            int sum = 0;
            // now, count
            for (int x = 0; x < 5; x++)
                sum += bits(y, x) != ids[p][x];
            if (minSum > sum)
                minSum = sum;
        }
        // do the and
        dist += minSum;
    }

    return dist;
}

int analyzeMarkerImage(Mat& grey, int& nRotations) {

    // Markers  are divided in 7x7 regions, of which the inner 5x5 belongs to marker info
    // the external border shoould be entirely black

    int swidth = grey.rows / 7;
    for (int y = 0; y < 7; y++) {
        int inc = 6;
        if (y == 0 || y == 6)
            inc = 1; // for first and last row, check the whole border
        for (int x = 0; x < 7; x += inc) {
            int Xstart = (x) * (swidth);
            int Ystart = (y) * (swidth);
            Mat square = grey(Rect(Xstart, Ystart, swidth, swidth));
            int nZ = countNonZero(square);
            if (nZ > (swidth * swidth) / 2) {
                //      cout<<"neb"<<endl;
                return -1; // can not be a marker because the border element is not black!
            }
        }
    }

    // now,
    MarkerCode _bits;
    // get information(for each inner square, determine if it is  black or white)

    for (int y = 0; y < 5; y++) {

        for (int x = 0; x < 5; x++) {
            int Xstart = (x + 1) * (swidth);
            int Ystart = (y + 1) * (swidth);
            Mat square = grey(Rect(Xstart, Ystart, swidth, swidth));
            int nZ = countNonZero(square);
            if (nZ > (swidth * swidth) / 2)
                _bits(y, x) = 1;
        }
    }
    //      printMat<uchar>( _bits,"or mat");

    // checkl all possible rotations
    Mat _bitsFlip;
    MarkerCode Rotations[4];
    Rotations[0] = _bits;
    int minDist = hammDistMarker(Rotations[0]);

    for (int i = 1; i < 4; i++) {
        // rotate
        Rotations[i] = rotate(Rotations[i - 1]);
        // get the hamming distance to the nearest possible word
        int dist = hammDistMarker(Rotations[i]);
        if (dist < minDist) {
            minDist = dist;
            nRotations = i;
        }
    }
    //              printMat<uchar>( Rotations [ minDist.second]);
    //          cout<<"MinDist="<<minDist.first<<" "<<minDist.second<<endl;
    if (minDist != 0) // FUTURE WORK: correct if any error
        return -1;

    // Get id of the marker
    int MatID = 0;
    MarkerCode bits = Rotations[nRotations];
    for (int y = 0; y < 5; y++) {
        MatID <<= 1;
        if (bits(y, 1))
            MatID |= 1;
        MatID <<= 1;
        if (bits(y, 3))
            MatID |= 1;
    }
    return MatID;
}

bool correctHammMarker(const MarkerCode& bits) {
    // detect this lines with errors
    bool errors[4];
    int ids[4][5] = {{0, 0, 0, 0, 0}, {0, 0, 1, 1, 1}, {1, 1, 0, 0, 1}, {1, 1, 1, 1, 0}};

    for (int y = 0; y < 5; y++) {
        int minSum = 1e5;
        // hamming distance to each possible word
        for (int p = 0; p < 4; p++) {
            int sum = 0;
            // now, count
            for (int x = 0; x < 5; x++)
                sum += bits(y, x) != ids[p][x];
            if (minSum > sum)
                minSum = sum;
        }

        errors[y] = minSum != 0;
    }

    return true;
}
}

namespace aruco {

/************************************
 *
 *
 *
 *
 ************************************/
/**
*/
Mat FiducidalMarkers::createMarkerImage(int id, int size, bool addWaterMark, bool locked) {
    CV_Assert(0 <= id && id < 1024);

    Mat_<uchar> marker(size, size, uchar(0));

    // for each line, create
    int swidth = size / 7;
    int ids[4] = {0x10, 0x17, 0x09, 0x0e};
    for (int y = 0; y < 5; y++) {
        int index = (id >> 2 * (4 - y)) & 0x0003;
        int val = ids[index];
        for (int x = 0; x < 5; x++) {
            if ((val >> (4 - x)) & 0x0001)
                marker(Rect((x + 1) * swidth, (y + 1) * swidth, swidth, swidth)) = 255;
        }
    }

    if (addWaterMark) {
        char idcad[30];
        sprintf(idcad, "#%d", id);
        float ax = float(size) / 100.;
        int linew = 1 + (marker.rows / 500);
        cv::putText(marker, idcad, cv::Point(0, marker.rows - marker.rows / 40), cv::FONT_HERSHEY_COMPLEX,
                    ax * 0.15f, cv::Scalar::all(30), linew, CV_AA);
    }

    if (locked) {
        // add a locking
        int sqSize = float(size) * 0.25;

        cv::Mat_<uchar> lock_marker(size + sqSize * 2, size + sqSize * 2, uchar(255));
        // cerr<<lock_marker.size()<<endl;
        // write the squares
        lock_marker(cv::Range(0, sqSize), cv::Range(0, sqSize)) = 0;

        lock_marker(cv::Range(lock_marker.rows - sqSize, lock_marker.rows), cv::Range(0, sqSize)) = 0;

        lock_marker(cv::Range(lock_marker.rows - sqSize, lock_marker.rows),
                          cv::Range(lock_marker.cols - sqSize, lock_marker.cols)) = 0;

        lock_marker(cv::Range(0, sqSize), cv::Range(lock_marker.cols - sqSize, lock_marker.cols)) = 0;

        marker.copyTo(lock_marker(Range(sqSize, marker.rows + sqSize), Range(sqSize, marker.cols + sqSize)));
        marker = lock_marker;
    }
    return marker;
}
/**
 *
 */
cv::Mat FiducidalMarkers::getMarkerMat(int id){
    CV_Assert(0 <= id && id < 1024 && "Invalid marker id");

    Mat_<uchar> marker(5, 5, uchar(0));

    // for each line, create
    int ids[4] = {0x10, 0x17, 0x09, 0x0e};
    for (int y = 0; y < 5; y++) {
        int index = (id >> 2 * (4 - y)) & 0x0003;
        int val = ids[index];
        for (int x = 0; x < 5; x++) {
            if ((val >> (4 - x)) & 0x0001)
                marker(y, x) = 1;
            else
                marker(y, x) = 0;
        }
    }
    return marker;
}
/************************************
 *
 *
 *
 *
 ************************************/

cv::Mat FiducidalMarkers::createBoardImage(Size gridSize, int MarkerSize, int MarkerDistance, BoardConfiguration& TInfo,
                                           const vector<int>& excludedIds) {
    int nMarkers = gridSize.height * gridSize.width;
    TInfo.resize(nMarkers);
    vector<int> ids = getListOfValidMarkersIds_random(nMarkers, excludedIds);
    for (int i = 0; i < nMarkers; i++)
        TInfo[i].id = ids[i];

    int sizeY = gridSize.height * MarkerSize + (gridSize.height - 1) * MarkerDistance;
    int sizeX = gridSize.width * MarkerSize + (gridSize.width - 1) * MarkerDistance;
    // find the center so that the ref systeem is in it
    int centerX = sizeX / 2;
    int centerY = sizeY / 2;

    // indicate the data is expressed in pixels
    TInfo.mInfoType = BoardConfiguration::PIX;
    Mat tableImage(sizeY, sizeX, CV_8UC1);
    tableImage.setTo(Scalar(255));
    int idp = 0;
    for (int y = 0; y < gridSize.height; y++)
        for (int x = 0; x < gridSize.width; x++, idp++) {
            Mat subrect(tableImage, Rect(x * (MarkerDistance + MarkerSize),
                                         y * (MarkerDistance + MarkerSize), MarkerSize, MarkerSize));
            Mat marker = createMarkerImage(TInfo[idp].id, MarkerSize);
            // set the location of the corners
            TInfo[idp].resize(4);
            TInfo[idp][0] =
                    cv::Point3f(x * (MarkerDistance + MarkerSize), y * (MarkerDistance + MarkerSize), 0);
            TInfo[idp][1] = cv::Point3f(x * (MarkerDistance + MarkerSize) + MarkerSize,
                                        y * (MarkerDistance + MarkerSize), 0);
            TInfo[idp][2] = cv::Point3f(x * (MarkerDistance + MarkerSize) + MarkerSize,
                                        y * (MarkerDistance + MarkerSize) + MarkerSize, 0);
            TInfo[idp][3] = cv::Point3f(x * (MarkerDistance + MarkerSize),
                                        y * (MarkerDistance + MarkerSize) + MarkerSize, 0);
            for (int i = 0; i < 4; i++)
                TInfo[idp][i] -= cv::Point3f(centerX, centerY, 0);
            marker.copyTo(subrect);
        }

    return tableImage;
}

/************************************
 *
 *
 *
 *
 ************************************/
cv::Mat FiducidalMarkers::createBoardImage_ChessBoard(Size gridSize, int MarkerSize, BoardConfiguration& TInfo, bool centerData,
                                                      const vector<int>& excludedIds){
    // determine the total number of markers required
    int nMarkers = 3 * (gridSize.width * gridSize.height) / 4; // overdetermine  the number of marker read
    vector<int> idsVector = getListOfValidMarkersIds_random(nMarkers, excludedIds);

    int sizeY = gridSize.height * MarkerSize;
    int sizeX = gridSize.width * MarkerSize;
    // find the center so that the ref systeem is in it
    int centerX = sizeX / 2;
    int centerY = sizeY / 2;

    Mat tableImage(sizeY, sizeX, CV_8UC1);
    tableImage.setTo(Scalar(255));
    TInfo.mInfoType = BoardConfiguration::PIX;
    int CurMarkerIdx = 0;
    for (int y = 0; y < gridSize.height; y++) {

        bool toWrite;
        if (y % 2 == 0)
            toWrite = false;
        else
            toWrite = true;
        for (int x = 0; x < gridSize.width; x++) {
            toWrite = !toWrite;
            if (toWrite) {
                CV_Assert(CurMarkerIdx < idsVector.size() && "INTERNAL ERROR. REWRITE THIS!!");
                TInfo.push_back(MarkerInfo(idsVector[CurMarkerIdx++]));

                Mat subrect(tableImage, Rect(x * MarkerSize, y * MarkerSize, MarkerSize, MarkerSize));
                Mat marker = createMarkerImage(TInfo.back().id, MarkerSize);
                // set the location of the corners
                TInfo.back().resize(4);
                TInfo.back()[0] = cv::Point3f(x * (MarkerSize), y * (MarkerSize), 0);
                TInfo.back()[1] = cv::Point3f(x * (MarkerSize) + MarkerSize, y * (MarkerSize), 0);
                TInfo.back()[2] =
                        cv::Point3f(x * (MarkerSize) + MarkerSize, y * (MarkerSize) + MarkerSize, 0);
                TInfo.back()[3] = cv::Point3f(x * (MarkerSize), y * (MarkerSize) + MarkerSize, 0);
                if (centerData) {
                    for (int i = 0; i < 4; i++)
                        TInfo.back()[i] -= cv::Point3f(centerX, centerY, 0);
                }
                marker.copyTo(subrect);
            }
        }
    }

    return tableImage;
}

/************************************
 *
 *
 *
 *
 ************************************/
cv::Mat FiducidalMarkers::createBoardImage_Frame(Size gridSize, int MarkerSize, int MarkerDistance, BoardConfiguration& TInfo,
                                                 bool centerData, const vector<int>& excludedIds) {
    int nMarkers = 2 * gridSize.height * 2 * gridSize.width;
    vector<int> idsVector = getListOfValidMarkersIds_random(nMarkers, excludedIds);

    int sizeY = gridSize.height * MarkerSize + MarkerDistance * (gridSize.height - 1);
    int sizeX = gridSize.width * MarkerSize + MarkerDistance * (gridSize.width - 1);
    // find the center so that the ref systeem is in it
    int centerX = sizeX / 2;
    int centerY = sizeY / 2;

    Mat tableImage(sizeY, sizeX, CV_8UC1);
    tableImage.setTo(Scalar(255));
    TInfo.mInfoType = BoardConfiguration::PIX;
    int CurMarkerIdx = 0;
    int mSize = MarkerSize + MarkerDistance;
    for (int y = 0; y < gridSize.height; y++) {
        for (int x = 0; x < gridSize.width; x++) {
            if (y == 0 || y == gridSize.height - 1 || x == 0 || x == gridSize.width - 1) {
                TInfo.push_back(MarkerInfo(idsVector[CurMarkerIdx++]));
                Mat subrect(tableImage, Rect(x * mSize, y * mSize, MarkerSize, MarkerSize));
                Mat marker = createMarkerImage(TInfo.back().id, MarkerSize);
                marker.copyTo(subrect);
                // set the location of the corners
                TInfo.back().resize(4);
                TInfo.back()[0] = cv::Point3f(x * (mSize), y * (mSize), 0);
                TInfo.back()[1] = cv::Point3f(x * (mSize) + MarkerSize, y * (mSize), 0);
                TInfo.back()[2] = cv::Point3f(x * (mSize) + MarkerSize, y * (mSize) + MarkerSize, 0);
                TInfo.back()[3] = cv::Point3f(x * (mSize), y * (mSize) + MarkerSize, 0);
                if (centerData) {
                    for (int i = 0; i < 4; i++)
                        TInfo.back()[i] -= cv::Point3f(centerX, centerY, 0);
                }
            }
        }
    }

    return tableImage;
}

/************************************
 *
 *
 *
 *
 ************************************/
int FiducidalMarkers::detect(const Mat& in, int& nRotations) {
    assert(in.rows == in.cols);
    Mat grey;
    if (in.type() == CV_8UC1)
        grey = in;
    else
        cv::cvtColor(in, grey, CV_BGR2GRAY);
    // threshold image
    threshold(grey, grey, 125, 255, THRESH_BINARY | THRESH_OTSU);

    // now, analyze the interior in order to get the id
    // try first with the big ones

    return analyzeMarkerImage(grey, nRotations);
}
}
