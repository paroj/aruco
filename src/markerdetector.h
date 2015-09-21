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

#ifndef _ARUCO_MarkerDetector_H
#define _ARUCO_MarkerDetector_H

#include <opencv2/core/core.hpp>

#include "aruco_export.h"
#include "cameraparameters.h"
#include "marker.h"

namespace aruco {

/**\brief Main class for marker detection
 *
 */
class ARUCO_EXPORTS MarkerDetector {
    // Represent a candidate to be a maker
    class MarkerCandidate : public Marker {
    public:
        MarkerCandidate() {}
        MarkerCandidate(const Marker& M) : Marker(M) {}
        MarkerCandidate(const MarkerCandidate& M) : Marker(M) {
            contour = M.contour;
            idx = M.idx;
        }
        MarkerCandidate& operator=(const MarkerCandidate& M) {
            (*(Marker*)this) = (*(Marker*)&M);
            contour = M.contour;
            idx = M.idx;
            return *this;
        }

        std::vector<cv::Point> contour; // all the points of its contour
        int idx;                   // index position in the global contour list
    };

public:
    /**
     * The function that identifies a marker
     *
     * The marker function receives the image 'in' with the region that might contain one of your markers.
     * These are the rectangular regions with black in the image.
     *
     * As output your marker function must indicate the following information. First, the output parameter
     * nRotations must indicate how many times the marker
     * must be rotated clockwise 90 deg  to be in its ideal position. (The way you would see it when you
     * print it). This is employed to know
     * always which is the corner that acts as reference system. Second, the function must return -1 if the
     * image does not contains one of your markers, and its id otherwise.
     */
    typedef int (*MarkerdetectorFunc)(const cv::Mat& in, int& nRotations);

    /**
     * See
     */
    MarkerDetector();

    /**
     */
    ~MarkerDetector();

    /**Detects the markers in the image passed
     *
     * If you provide information about the camera parameters and the size of the marker, then, the
     *extrinsics of the markers are detected
     *
     * @param input input color image
     * @param detectedMarkers output vector with the markers detected
     * @param camMatrix intrinsic camera information.
     * @param distCoeff camera distorsion coefficient. If set Mat() if is assumed no camera distorion
     * @param markerSizeMeters size of the marker sides expressed in meters
     * @param setYPerpendicular If set the Y axis will be perpendicular to the surface. Otherwise, it will
     *be the Z axis
     */
    void detect(cv::InputArray input, std::vector<Marker>& detectedMarkers, cv::Mat camMatrix = cv::Mat(),
                cv::Mat distCoeff = cv::Mat(), float markerSizeMeters = -1, bool setYPerpendicular = false) ;
    /**Detects the markers in the image passed
     *
     * If you provide information about the camera parameters and the size of the marker, then, the
     *extrinsics of the markers are detected
     *
     * @param input input color image
     * @param detectedMarkers output vector with the markers detected
     * @param camParams Camera parameters
     * @param markerSizeMeters size of the marker sides expressed in meters
     * @param setYPerpendicular If set the Y axis will be perpendicular to the surface. Otherwise, it will
     *be the Z axis
     */
    void detect(cv::InputArray input, std::vector<Marker>& detectedMarkers, const CameraParameters& camParams,
                float markerSizeMeters = -1, bool setYPerpendicular = false) {
        detect(input, detectedMarkers, camParams.CameraMatrix, camParams.Distorsion, markerSizeMeters,
               setYPerpendicular);
    }

    /**This set the type of thresholding methods available
     */

    enum ThresholdMethods { FIXED_THRES, ADPT_THRES, CANNY };

    /**Sets the threshold method
     */
    void setThresholdMethod(ThresholdMethods m) { _thresMethod = m; }
    /**Returns the current threshold method
     */
    ThresholdMethods getThresholdMethod() const { return _thresMethod; }
    /**
     * Set the parameters of the threshold method
     * We are currently using the Adptive threshold ee opencv doc of adaptiveThreshold for more info
     *   @param param1: blockSize of the pixel neighborhood that is used to calculate a threshold value for
     * the pixel
     *   @param param2: The constant subtracted from the mean or weighted mean
     */
    void setThresholdParams(double param1, double param2) {
        _thresParam1 = param1;
        _thresParam2 = param2;
    }

    /**Allows for a parallel search of several values of the param1 simultaneously (in different threads)
     * The param r1 the indicates how many values around the current value of param1 are evaluated. In other
     *words
     * if r1>0, param1 is searched in range [param1- r1 ,param1+ r1 ]
     *
     * r2 unused yet. Added in case of future need.
     */
    void setThresholdParamRange(size_t r1 = 0, size_t r2 = 0) { _thresParam1_range = r1; }
    /**
     * This method assumes that the markers may have some of its corners joined either to another marker
     * in a chessboard like pattern) or to a rectangle. This is the case in which the subpixel refinement
     * method in opencv work best.
     *
     * Enabling this does not force you to use locked corners, normals markers will be detected also.
     *However,
     * when using locked corners, enabling this option will increase robustness in detection at the cost of
     * higher computational time.
     * ,
     * Note for developer: Enabling this option forces a call to findCornerMaxima
     */
    void enableLockedCornersMethod(bool enable);

    /**
     * Set the parameters of the threshold method
     * We are currently using the Adptive threshold ee opencv doc of adaptiveThreshold for more info
     *   param1: blockSize of the pixel neighborhood that is used to calculate a threshold value for the
     * pixel
     *   param2: The constant subtracted from the mean or weighted mean
     */
    void getThresholdParams(double& param1, double& param2) const {
        param1 = _thresParam1;
        param2 = _thresParam2;
    }

    /**Returns a reference to the internal image thresholded. It is for visualization purposes and to adjust
     * manually
     * the parameters
     */
    const cv::Mat& getThresholdedImage() { return thres; }
    /**Methods for corner refinement
     */
    enum CornerRefinementMethod { NONE, HARRIS, SUBPIX, LINES };
    /**
     */
    void setCornerRefinementMethod(CornerRefinementMethod method) { _cornerMethod = method; }
    /**
     */
    CornerRefinementMethod getCornerRefinementMethod() const { return _cornerMethod; }
    /**Specifies the min and max sizes of the markers as a fraction of the image size. By size we mean the
     *maximum
     * of cols and rows.
     * @param min size of the contour to consider a possible marker as valid (0,1]
     * @param max size of the contour to consider a possible marker as valid [0,1)
     *
     */
    void setMinMaxSize(float min = 0.03, float max = 0.5);

    /**reads the min and max sizes employed
     * @param min output size of the contour to consider a possible marker as valid (0,1]
     * @param max output size of the contour to consider a possible marker as valid [0,1)
     *
     */
    void getMinMaxSize(float& min, float& max) {
        min = _minSize;
        max = _maxSize;
    }

    /**
     * Specifies a value to indicate the required speed for the internal processes. If you need maximum
     *speed (at the cost of a lower detection rate),
     * use the value 3, If you rather a more precise and slow detection, set it to 0.
     *
     * Actually, the main differences are that in highspeed mode, we employ setCornerRefinementMethod(NONE)
     *and internally, we use a small canonical
     * image to detect the marker. In low speed mode, we use setCornerRefinementMethod(HARRIS) and a bigger
     *size for the canonical marker image
     */
    void setDesiredSpeed(int val);
    /**
     */
    int getDesiredSpeed() const { return _speed; }

    /**
     * Specifies the size for the canonical marker image. A big value makes the detection slower than a
     * small value.
     * Minimun value is 10. Default value is 56.
     */
    void setWarpSize(int val);

    /**
     */
    int getWarpSize() const { return _markerWarpSize; }

    /**
     * Allows to specify the function that identifies a marker. Therefore, you can create your own type of
     * markers different from these
     * employed by default in the library.
     */
    void setMakerDetectorFunction(MarkerdetectorFunc markerdetector_func) {
        markerIdDetectorFunc = markerdetector_func;
    }

    ///-------------------------------------------------
    /// Methods you may not need
    /// Thesde methods do the hard work. They have been set public in case you want to do customizations
    ///-------------------------------------------------

    /**
     * Thesholds the passed image with the specified method.
     */
    void thresHold(int method, cv::InputArray grey, cv::OutputArray thresImg, double param1 = -1,
                   double param2 = -1);
    /**
    * Detection of candidates to be markers, i.e., rectangles.
    * This function returns in candidates all the rectangles found in a thresolded image
    */
    void detectRectangles(cv::InputArray thresImg, std::vector<std::vector<cv::Point2f> >& candidates);

    /**Returns a list candidates to be markers (rectangles), for which no valid id was found after calling
     * detectRectangles
     */
    const std::vector<std::vector<cv::Point2f> >& getCandidates() { return _candidates; }

    /**Given the iput image with markers, creates an output image with it in the canonical position
     * @param in input image
     * @param out image with the marker
     * @param size of out
     * @param points 4 corners of the marker in the image in
     * @return true if the operation succeed
     */
    void warp(cv::InputArray in, cv::OutputArray out, cv::Size size, std::vector<cv::Point2f> points) ;

    /** Refine MarkerCandidate Corner using LINES method
     * @param candidate candidate to refine corners
     */
    void refineCandidateLines(MarkerCandidate& candidate, const cv::Mat& camMatrix, const cv::Mat& distCoeff);

private:
    bool warp_cylinder(cv::Mat& in, cv::Mat& out, cv::Size size, MarkerCandidate& mc);
    /**
    * Detection of candidates to be markers, i.e., rectangles.
    * This function returns in candidates all the rectangles found in a thresolded image
    */
    void detectRectangles(std::vector<cv::Mat>& vimages, std::vector<MarkerCandidate>& candidates);
    // Current threshold method
    ThresholdMethods _thresMethod;
    // Threshold parameters
    double _thresParam1, _thresParam2, _thresParam1_range;
    // Current corner method
    CornerRefinementMethod _cornerMethod;
    // minimum and maximum size of a contour lenght
    float _minSize, _maxSize;

    // is corner locked
    bool _useLockedCorners;

    // Speed control
    int _speed;
    int _markerWarpSize;
    float _borderDistThres; // border around image limits in which corners are not allowed to be detected.
    // vectr of candidates to be markers. This is a vector with a set of rectangles that have no valid id
    std::vector<std::vector<cv::Point2f> > _candidates;
    // Images
    cv::Mat thres;
    // pointer to the function that analizes a rectangular region so as to detect its internal marker
    MarkerdetectorFunc markerIdDetectorFunc;
};
}
#endif
