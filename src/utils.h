/*
 * utils.h
 *
 *  Created on: 23.07.2015
 *      Author: parojtbe
 */

#ifndef aruco_UTILS_HPP
#define aruco_UTILS_HPP

#include <aruco_export.h>
#include <opencv2/core.hpp>

namespace aruco {

/**
 * change rotation such that Y axis points up
 */
ARUCO_EXPORTS void rotateXAxis(cv::Vec3f& rotation);

/**Given the extrinsic camera parameters returns the GL_MODELVIEW matrix for opengl.
 * Setting this matrix, the reference coordinate system will be set in this marker
 */
ARUCO_EXPORTS void GetGLModelViewMatrix(const cv::Vec3f& Rvec, const cv::Vec3f& Tvec, double modelview_matrix[16]);

/**
 * Returns position vector and orientation quaternion for an Ogre scene node or entity.
 *  Use:
 * ...
 * Ogre::Vector3 ogrePos (position[0], position[1], position[2]);
 * Ogre::Quaternion  ogreOrient (orientation[0], orientation[1], orientation[2], orientation[3]);
 * mySceneNode->setPosition( ogrePos  );
 * mySceneNode->setOrientation( ogreOrient  );
 * ...
 */
ARUCO_EXPORTS void GetOgrePoseParameters(const cv::Vec3f& Rvec, const cv::Vec3f& Tvec, double position[3], double orientation[4]);

static inline float perimeter(const std::vector<cv::Point2f>& a) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); i++) {
        size_t i2 = (i + 1) % a.size();
        sum += norm(a[i] - a[i2]);
    }
    return sum;
}
} /* namespace aruco */
#endif
