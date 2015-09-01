/*
 * utils.cpp
 *
 *  Created on: 23.07.2015
 *      Author: parojtbe
 */

#include "utils.h"

#include <opencv2/calib3d.hpp>

using namespace cv;

namespace aruco {

void rotateXAxis(Mat& rotation) {
    cv::Matx33f R;
    Rodrigues(rotation, R);
    // create a rotation matrix for x axis
    cv::Matx33f RX = cv::Matx33f::eye();
    float angleRad = CV_PI / 2;
    RX(1, 1) = cos(angleRad);
    RX(1, 2) = -sin(angleRad);
    RX(2, 1) = sin(angleRad);
    RX(2, 2) = cos(angleRad);
    // now multiply
    R = R * RX;
    // finally, the the rodrigues back
    Rodrigues(R, rotation);
}

void GetGLModelViewMatrix(const Mat_<double>& Rvec, const Mat_<double>& Tvec, double modelview_matrix[16]) {
    CV_Assert(!Rvec.empty() && !Tvec.empty() && "extrinsic parameters are not set");

    Matx33d Rot;
    Rodrigues(Rvec, Rot);

    double para[3][4];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            para[i][j] = Rot(i, j);
    // now, add the translation
    para[0][3] = Tvec(0);
    para[1][3] = Tvec(1);
    para[2][3] = Tvec(2);

    modelview_matrix[0 + 0 * 4] = para[0][0];
    // R1C2
    modelview_matrix[0 + 1 * 4] = para[0][1];
    modelview_matrix[0 + 2 * 4] = para[0][2];
    modelview_matrix[0 + 3 * 4] = para[0][3];
    // R2
    modelview_matrix[1 + 0 * 4] = para[1][0];
    modelview_matrix[1 + 1 * 4] = para[1][1];
    modelview_matrix[1 + 2 * 4] = para[1][2];
    modelview_matrix[1 + 3 * 4] = para[1][3];
    // R3
    modelview_matrix[2 + 0 * 4] = -para[2][0];
    modelview_matrix[2 + 1 * 4] = -para[2][1];
    modelview_matrix[2 + 2 * 4] = -para[2][2];
    modelview_matrix[2 + 3 * 4] = -para[2][3];
    modelview_matrix[3 + 0 * 4] = 0.0;
    modelview_matrix[3 + 1 * 4] = 0.0;
    modelview_matrix[3 + 2 * 4] = 0.0;
    modelview_matrix[3 + 3 * 4] = 1.0;
/*
    double scale = 1;
    if (scale != 1.0) {
        modelview_matrix[12] *= scale;
        modelview_matrix[13] *= scale;
        modelview_matrix[14] *= scale;
    } */
}

void GetOgrePoseParameters(const Mat_<double>& Rvec, const Mat_<double>& Tvec, double position[3], double orientation[4]) {
    CV_Assert(!Rvec.empty() && !Tvec.empty() && "extrinsic parameters are not set");

    // calculate position vector
    position[0] = -Tvec(0);
    position[1] = -Tvec(1);
    position[2] = +Tvec(2);

    // now calculare orientation quaternion
    cv::Matx33d Rot;
    cv::Rodrigues(Rvec, Rot);

    // calculate axes for quaternion
    double stAxes[3][3];
    // x axis
    stAxes[0][0] = -Rot(0, 0);
    stAxes[0][1] = -Rot(1, 0);
    stAxes[0][2] = +Rot(2, 0);
    // y axis
    stAxes[1][0] = -Rot(0, 1);
    stAxes[1][1] = -Rot(1, 1);
    stAxes[1][2] = +Rot(2, 1);
    // for z axis, we use cross product
    stAxes[2][0] = stAxes[0][1] * stAxes[1][2] - stAxes[0][2] * stAxes[1][1];
    stAxes[2][1] = -stAxes[0][0] * stAxes[1][2] + stAxes[0][2] * stAxes[1][0];
    stAxes[2][2] = stAxes[0][0] * stAxes[1][1] - stAxes[0][1] * stAxes[1][0];

    // transposed matrix
    double axes[3][3];
    axes[0][0] = stAxes[0][0];
    axes[1][0] = stAxes[0][1];
    axes[2][0] = stAxes[0][2];

    axes[0][1] = stAxes[1][0];
    axes[1][1] = stAxes[1][1];
    axes[2][1] = stAxes[1][2];

    axes[0][2] = stAxes[2][0];
    axes[1][2] = stAxes[2][1];
    axes[2][2] = stAxes[2][2];

    // Algorithm in Ken Shoemake's article in 1987 SIGGRAPH course notes
    // article "Quaternion Calculus and Fast Animation".
    double fTrace = axes[0][0] + axes[1][1] + axes[2][2];
    double fRoot;

    if (fTrace > 0.0) {
        // |w| > 1/2, may as well choose w > 1/2
        fRoot = sqrt(fTrace + 1.0); // 2w
        orientation[0] = 0.5 * fRoot;
        fRoot = 0.5 / fRoot; // 1/(4w)
        orientation[1] = (axes[2][1] - axes[1][2]) * fRoot;
        orientation[2] = (axes[0][2] - axes[2][0]) * fRoot;
        orientation[3] = (axes[1][0] - axes[0][1]) * fRoot;
    } else {
        // |w| <= 1/2
        static unsigned int s_iNext[3] = {1, 2, 0};
        unsigned int i = 0;
        if (axes[1][1] > axes[0][0])
            i = 1;
        if (axes[2][2] > axes[i][i])
            i = 2;
        unsigned int j = s_iNext[i];
        unsigned int k = s_iNext[j];

        fRoot = sqrt(axes[i][i] - axes[j][j] - axes[k][k] + 1.0);
        double* apkQuat[3] = {&orientation[1], &orientation[2], &orientation[3]};
        *apkQuat[i] = 0.5 * fRoot;
        fRoot = 0.5 / fRoot;
        orientation[0] = (axes[k][j] - axes[j][k]) * fRoot;
        *apkQuat[j] = (axes[j][i] + axes[i][j]) * fRoot;
        *apkQuat[k] = (axes[k][i] + axes[i][k]) * fRoot;
    }
}

} /* namespace aruco */
