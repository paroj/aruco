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

#include "highlyreliablemarkers.h"
#include <opencv2/imgproc/imgproc.hpp>

#include "arucofidmarkers.h"
#include "serialization.h"

using namespace std;
using namespace cv;

namespace aruco {
namespace {
// convert to string
template <class T> static string to_string(T num) {
    stringstream ss;
    ss << num;
    return ss.str();
}

/**
 * Return hamming distance between two bit vectors
 */
unsigned int hammingDistance(const vector<bool>& m1, const vector<bool>& m2) {
    unsigned int res = 0;
    for (unsigned int i = 0; i < m1.size(); i++)
        if (m1[i] != m2[i])
            res++;
    return res;
}

class MarkerGenerator {

private:
    int _nTransitions;
    std::vector<int> _transitionsWeigth;
    int _totalWeigth;
    int _n;

public:
    MarkerGenerator(int n) {
        _n = n;
        _nTransitions = n - 1;
        _transitionsWeigth.resize(_nTransitions);
        _totalWeigth = 0;
        for (int i = 0; i < _nTransitions; i++) {
            _transitionsWeigth[i] = i;
            _totalWeigth += i;
        }
    }

    MarkerCode generateMarker() {
        Mat_<uchar> code(_n, _n);

        for (int w = 0; w < _n; w++) {
            Mat_<uchar> currentWord = code.row(w);

            int randomNum = rand() % _totalWeigth;
            int currentNTransitions = _nTransitions - 1;
            for (int k = 0; k < _nTransitions; k++) {
                if (_transitionsWeigth[k] > randomNum) {
                    currentNTransitions = k;
                    break;
                }
            }
            std::vector<int> transitionsIndexes(_nTransitions);
            for (int i = 0; i < _nTransitions; i++)
                transitionsIndexes[i] = i;
            std::random_shuffle(transitionsIndexes.begin(), transitionsIndexes.end());

            std::vector<int> selectedIndexes;
            for (int k = 0; k < currentNTransitions; k++)
                selectedIndexes.push_back(transitionsIndexes[k]);
            std::sort(selectedIndexes.begin(), selectedIndexes.end());
            int currBit = rand() % 2;
            size_t currSelectedIndexesIdx = 0;
            for (int k = 0; k < _n; k++) {
                currentWord(k) = currBit;
                if (currSelectedIndexesIdx < selectedIndexes.size() &&
                    k == selectedIndexes[currSelectedIndexesIdx]) {
                    currBit = 1 - currBit;
                    currSelectedIndexesIdx++;
                }
            }
        }

        MarkerCode emptyMarker(_n);
        emptyMarker.set(code);
        return emptyMarker;
    }
};
}

// static variables from HighlyReliableMarkers. Need to be here to avoid linking errors
Dictionary HighlyReliableMarkers::_D;
HighlyReliableMarkers::BalancedBinaryTree HighlyReliableMarkers::_binaryTree;
unsigned int HighlyReliableMarkers::_n, HighlyReliableMarkers::_ncellsBorder,
    HighlyReliableMarkers::_correctionDistance;

/**
*/
MarkerCode::MarkerCode(unsigned int n) {
    // resize bits vectors and initialize to 0
    for (unsigned int i = 0; i < 4; i++) {
        _bits[i].resize(n * n);
        for (unsigned int j = 0; j < _bits[i].size(); j++)
            _bits[i][j] = 0;
        _ids[i] = 0; // ids are also 0
    }
    _n = n;
};

/**
 */
MarkerCode::MarkerCode(const MarkerCode& MC) {
    for (unsigned int i = 0; i < 4; i++) {
        _bits[i] = MC._bits[i];
        _ids[i] = MC._ids[i];
    }
    _n = MC._n;
}

void MarkerCode::set(const cv::Mat& _code) {
    Mat_<uchar> code(_code);

    for(size_t y = 0; y < _n; y++) {
        for(size_t x = 0; x < _n; x++) {
            for (size_t i = 0; i < 4; i++) {         // calculate bit coordinates for each rotation
                size_t _x = x, _y = y;

                // if rotation 0, dont do anything
                // else calculate bit position in that rotation
                if (i == 1) {
                    _y = x;
                    _x = n() - y - 1;
                } else if (i == 2) {
                    _y = n() - y - 1;
                    _x = n() - x - 1;
                } else if (i == 3) {
                    _y = n() - x - 1;
                    _x = y;
                }

                size_t rotPos = _y * n() + _x;     // calculate position in the unidimensional string

                bool val = code(y, x);
                _bits[i][rotPos] = val;            // modify value
                                                   // update identifier in that rotation
                if (val)
                    _ids[i] |= 2 << rotPos; // if 1, add 2^pos
            }
        }
    }
}

/**
 */
unsigned int MarkerCode::selfDistance(unsigned int& minRot) const {
    unsigned int res = _bits[0].size();    // init to n*n (max value)
    for (unsigned int i = 1; i < 4; i++) { // self distance is not calculated for rotation 0
        unsigned int hammdist = hammingDistance(_bits[0], _bits[i]);
        if (hammdist < res) {
            minRot = i;
            res = hammdist;
        }
    }
    return res;
}

/**
 */
unsigned int MarkerCode::distance(const MarkerCode& m, unsigned int& minRot) const {
    unsigned int res = _bits[0].size(); // init to n*n (max value)
    for (unsigned int i = 0; i < 4; i++) {
        unsigned int hammdist = hammingDistance(_bits[0], m.getRotation(i));
        if (hammdist < res) {
            minRot = i;
            res = hammdist;
        }
    }
    return res;
};

/**
 */
void MarkerCode::fromString(std::string s) {
    Mat_<char> code(_n, _n);
    for (unsigned int i = 0; i < s.length(); i++) {
        code(i/_n, i % _n) = s[i] == '1';
    }
    set(code);
}

/**
 */
std::string MarkerCode::toString() const {
    std::string s(size(), '0');

    for (unsigned int i = 0; i < size(); i++) {
        if (get(i))
            s[i] = '1';
    }
    return s;
}

/**
 */
cv::Mat MarkerCode::getImg(unsigned int pixSize) const {
    const unsigned int borderSize = 1;
    unsigned int nrows = n() + 2 * borderSize;
    if (pixSize % nrows != 0)
        pixSize = pixSize + nrows - pixSize % nrows;
    unsigned int cellSize = pixSize / nrows;
    cv::Mat img(pixSize, pixSize, CV_8U, cv::Scalar::all(0)); // create black image (init image to 0s)
    // double for to go over all the cells
    for (unsigned int i = 0; i < n(); i++) {
        for (unsigned int j = 0; j < n(); j++) {
            if (_bits[0][i * n() + j] != 0) { // just draw if it is 1, since the image has been init to 0
                                              // double for to go over all the pixels in the cell
                for (unsigned int k = 0; k < cellSize; k++) {
                    for (unsigned int l = 0; l < cellSize; l++) {
                        img.at<uchar>((i + borderSize) * cellSize + k, (j + borderSize) * cellSize + l) =
                            255;
                    }
                }
            }
        }
    }
    return img;
}

/**
*/
bool Dictionary::fromFile(std::string filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    fs.root() >> *this;
    return true;
};

/**
 */
bool Dictionary::toFile(std::string filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    // save number of markers
    fs << *this;
    return true;
};

/**
 */
unsigned int Dictionary::distance(const MarkerCode& m, unsigned int& minMarker, unsigned int& minRot) {
    unsigned int res = m.size();
    for (unsigned int i = 0; i < size(); i++) {
        unsigned int minRotAux;
        unsigned int distance = (*this)[i].distance(m, minRotAux);
        if (distance < res) {
            minMarker = i;
            minRot = minRotAux;
            res = distance;
        }
    }
    return res;
}

/**
 */
unsigned int Dictionary::minimunDistance() {
    if (size() == 0)
        return 0;
    unsigned int minDist = (*this)[0].size();
    // for each marker in D
    for (unsigned int i = 0; i < size(); i++) {
        // calculate self distance of the marker
        minDist = std::min(minDist, (*this)[i].selfDistance());

        // calculate distance to all the following markers
        for (unsigned int j = i + 1; j < size(); j++) {
            minDist = std::min(minDist, (*this)[i].distance((*this)[j]));
        }
    }
    return minDist;
}

/**
 */
bool HighlyReliableMarkers::loadDictionary(Dictionary D, float correctionDistanceRate) {
    if (D.size() == 0)
        return false;
    _D = D;
    _n = _D[0].n();
    _ncellsBorder = (_D[0].n() + 2);
    _correctionDistance = correctionDistanceRate * ((D.tau0 - 1) / 2);
    cerr << "aruco :: _correctionDistance = " << _correctionDistance << endl;
    _binaryTree.loadDictionary(D);
    return true;
}

bool HighlyReliableMarkers::loadDictionary(std::string filename, float correctionDistance) {
    Dictionary D;
    D.fromFile(filename);
    return loadDictionary(D, correctionDistance);
}

/**
 */
int HighlyReliableMarkers::detect(const cv::Mat& in, int& nRotations) {

    assert(in.rows == in.cols);
    cv::Mat grey;
    if (in.type() == CV_8UC1)
        grey = in;
    else
        cv::cvtColor(in, grey, CV_BGR2GRAY);
    // threshold image
    cv::threshold(grey, grey, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    int cellSize = grey.rows / _ncellsBorder;

    // check borders, even not necesary for the highly reliable markers
    // if(!checkBorders(grey, _ncellsBorder, cellSize)) return -1;

    // obtain inner code
    MarkerCode candidate(_n);
    candidate.set(getMarkerCode(grey, _n, cellSize));

#if 1
    // search each marker id in the balanced binary tree
    // has no advantage for dictionary with 100 markers
    unsigned int orgPos;
    for (unsigned int i = 0; i < 4; i++) {
        if (_binaryTree.findId(candidate.getId(i), orgPos)) {
            nRotations = i;
            // return candidate.getId(i);
            return orgPos;
        }
    }
#else
    // alternative version without using the balanced binary tree (less efficient)
    for(uint i=0; i<_D.size(); i++) {
        for(uint j=0; j<4; j++) {
            if(_D[i].getId() == candidate.getId(j)) {
                nRotations = j;
                return i;
            }
        }
    }
#endif

    // correct errors
    unsigned int minMarker, minRot;
    if (_D.distance(candidate, minMarker, minRot) <= _correctionDistance) {
        nRotations = minRot;
        return minMarker;
        // return _D[minMarker].getId();
    }

    return -1;
}

/**
 */
void HighlyReliableMarkers::BalancedBinaryTree::loadDictionary(Dictionary& D) {
    // create _orderD wich is a sorted version of D
    _orderD.clear();
    for (unsigned int i = 0; i < D.size(); i++) {
        _orderD.push_back(std::pair<unsigned int, unsigned int>(D[i].getId(), i));
    }
    std::sort(_orderD.begin(), _orderD.end());

    // calculate the number of levels of the tree
    unsigned int levels = 0;
    while (pow(float(2), float(levels)) <= _orderD.size())
        levels++;
    //       levels-=1; // only count full levels

    // auxiliar vector to know which elements are already in the tree
    std::vector<bool> visited;
    visited.resize(_orderD.size(), false);

    // calculate position of the root element
    unsigned int rootIdx = _orderD.size() / 2;
    visited[rootIdx] = true; // mark it as visited
    _root = rootIdx;

    //    for(int i=0; i<visited.size(); i++) std::cout << visited[i] << std::endl;

    // auxiliar vector to store the ids intervals (max and min) during the creation of the tree
    std::vector<std::pair<unsigned int, unsigned int> > intervals;
    // first, add the two intervals at each side of root element
    intervals.push_back(std::pair<unsigned int, unsigned int>(0, rootIdx));
    intervals.push_back(std::pair<unsigned int, unsigned int>(rootIdx, _orderD.size()));

    // init the tree
    _binaryTree.clear();
    _binaryTree.resize(_orderD.size());

    // add root information to the tree (make sure child do not coincide with self root for small sizes of
    // D)
    if (!visited[(0 + rootIdx) / 2])
        _binaryTree[rootIdx].first = (0 + rootIdx) / 2;
    else
        _binaryTree[rootIdx].first = -1;
    if (!visited[(rootIdx + _orderD.size()) / 2])
        _binaryTree[rootIdx].second = (rootIdx + _orderD.size()) / 2;
    else
        _binaryTree[rootIdx].second = -1;

    // for each tree level
    for (unsigned int i = 1; i < levels; i++) {
        unsigned int nintervals = intervals.size(); // count number of intervals and process them
        for (unsigned int j = 0; j < nintervals; j++) {
            // store interval information and delete it
            unsigned int lowerBound, higherBound;
            lowerBound = intervals.back().first;
            higherBound = intervals.back().second;
            intervals.pop_back();

            // center of the interval
            unsigned int center = (higherBound + lowerBound) / 2;

            // if center not visited, continue
            if (!visited[center])
                visited[center] = true;
            else
                continue;

            // calculate centers of the child intervals
            unsigned int lowerChild = (lowerBound + center) / 2;
            unsigned int higherChild = (center + higherBound) / 2;

            // if not visited (lower child)
            if (!visited[lowerChild]) {
                intervals.insert(intervals.begin(),
                                 std::pair<unsigned int, unsigned int>(
                                     lowerBound, center)); // add the interval to analyze later
                _binaryTree[center].first = lowerChild;    // add as a child in the tree
            } else
                _binaryTree[center].first = -1; // if not, mark as no child

            // (higher child, same as lower child)
            if (!visited[higherChild]) {
                intervals.insert(intervals.begin(),
                                 std::pair<unsigned int, unsigned int>(center, higherBound));
                _binaryTree[center].second = higherChild;
            } else
                _binaryTree[center].second = -1;
        }
    }

    // print tree
    //     for(uint i=0; i<_binaryTree.size(); i++) std::cout << _binaryTree[i].first << " " <<
    //     _binaryTree[i].second << std::endl;
    //     std::cout << std::endl;
}

/**
 */
bool HighlyReliableMarkers::BalancedBinaryTree::findId(unsigned int id, unsigned int& orgPos) {
    int pos = _root;                             // first position is root
    while (pos != -1) {                          // while having a valid position
        unsigned int posId = _orderD[pos].first; // calculate id of the node
        if (posId == id) {
            orgPos = _orderD[pos].second;
            return true; // if is the desire id, return true
        } else if (posId < id)
            pos = _binaryTree[pos].second; // if desired id is higher, look in higher child
        else
            pos = _binaryTree[pos].first; // if it is lower, look in lower child
    }
    return false; // if nothing found, return false
}

cv::Mat HighlyReliableMarkers::createBoardImage(cv::Size gridSize, const Dictionary& D, BoardConfiguration& BC,
                                                 bool chromatic) {
    unsigned int MarkerSize = (D[0].n() + 2) * 20;
    unsigned int MarkerDistance = MarkerSize / 5;

    int sizeY = gridSize.height * MarkerSize + (gridSize.height - 1) * MarkerDistance;
    int sizeX = gridSize.width * MarkerSize + (gridSize.width - 1) * MarkerDistance;
    // find the center so that the ref systeem is in it
    float centerX = sizeX / 2.;
    float centerY = sizeY / 2.;

    BC.mInfoType = BoardConfiguration::PIX;

    // indicate the data is expressed in pixels
    cv::Mat tableImage(sizeY, sizeX, CV_8UC1);
    tableImage.setTo(cv::Scalar(255));
    int idp = 0;
    for (int y = 0; y < gridSize.height; y++)
        for (int x = 0; x < gridSize.width; x++, idp += 1) {
            // create image
            cv::Mat subrect(tableImage,
                            cv::Rect(x * (MarkerDistance + MarkerSize), y * (MarkerDistance + MarkerSize),
                                     MarkerSize, MarkerSize));
            cv::Mat marker = D[idp].getImg(MarkerSize);
            marker.copyTo(subrect);

            // add to board configuration
            vector<Point3f> MI(4);
            BC.ids.push_back(D[idp].getId());
            for (unsigned int i = 0; i < 4; i++)
                MI[i].z = 0;
            MI[0].x = x * (MarkerDistance + MarkerSize) - centerX;
            MI[0].y = y * (MarkerDistance + MarkerSize) - centerY;
            MI[1].x = x * (MarkerDistance + MarkerSize) + MarkerSize - centerX;
            MI[1].y = y * (MarkerDistance + MarkerSize) - centerY;
            MI[2].x = x * (MarkerDistance + MarkerSize) + MarkerSize - centerX;
            MI[2].y = y * (MarkerDistance + MarkerSize) + MarkerSize - centerY;
            MI[3].x = x * (MarkerDistance + MarkerSize) - centerX;
            MI[3].y = y * (MarkerDistance + MarkerSize) + MarkerSize - centerY;
            // makes y negative so z axis is pointing up
            MI[0].y *= -1;
            MI[1].y *= -1;
            MI[2].y *= -1;
            MI[3].y *= -1;
            BC.objPoints.push_back(MI);
        }

    if (chromatic) {
        cv::Scalar color1 = cv::Scalar(250, 134, 4);
        //   cv::Scalar color2 = cv::Scalar(0,255,0);
        cv::Vec3b color2Vec3b = cv::Vec3b(0, 255, 0); // store as a Vec3b to assign easily to the image

        // create new image with border and with color 1
        cv::Mat chromaticImg(tableImage.rows + 2 * MarkerDistance, tableImage.cols + 2 * MarkerDistance,
                             CV_8UC3, color1);

        // now use color2 in black pixels
        for (int i = 0; i < tableImage.rows; i++) {
            for (int j = 0; j < tableImage.cols; j++) {
                if (tableImage.at<uchar>(i, j) == 0)
                    chromaticImg.at<cv::Vec3b>(MarkerDistance + i, MarkerDistance + j) = color2Vec3b;
            }
        }
        tableImage = chromaticImg;
    }

    return tableImage;
}

Dictionary HighlyReliableMarkers::createDicitionary(size_t dictSize, size_t n) {
    unsigned int tau = 2 * ((4 * ((n * n) / 4)) / 3);

    MarkerGenerator MG(n);

    const size_t MAX_UNPRODUCTIVE_ITERATIONS = 100000;
    int currentMaxUnproductiveIterations = MAX_UNPRODUCTIVE_ITERATIONS;

    unsigned int countUnproductive = 0;

    Dictionary D;
    while (D.size() < dictSize) {

        MarkerCode candidate;
        candidate = MG.generateMarker();

        if (candidate.selfDistance() >= tau && D.distance(candidate) >= tau) {
            D.push_back(candidate);
            countUnproductive = 0;
        } else {
            countUnproductive++;
            if (countUnproductive == currentMaxUnproductiveIterations) {
                tau--;
                countUnproductive = 0;
                //std::cout << "Reducing Tau to: " << tau << std::endl;

                if (tau == 0) {
                    CV_Error(CV_StsBadArg, "Error: Tau=0. Small marker size for too high number of markers. Stop");
                }

                if (D.size() >= 2)
                    currentMaxUnproductiveIterations = MAX_UNPRODUCTIVE_ITERATIONS;
                else
                    currentMaxUnproductiveIterations = MAX_UNPRODUCTIVE_ITERATIONS / 15;
            }
        }
    }

    D.tau0 = tau;

    return D;
}
}
