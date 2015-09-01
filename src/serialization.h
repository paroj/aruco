/*
 * serialization.h
 *
 *  Created on: 01.09.2015
 *      Author: parojtbe
 */

#ifndef _Aruco_serialization_h
#define _Aruco_serialization_h

#include "aruco.h"

namespace cv {

ARUCO_EXPORTS FileStorage& operator<<(FileStorage& fs, const aruco::Marker& m);
ARUCO_EXPORTS void read(const FileNode& ms, aruco::Marker& m,
                        const aruco::Marker& default_value = aruco::Marker());

ARUCO_EXPORTS FileStorage& operator<<(FileStorage& fs, const aruco::BoardConfiguration& m);
ARUCO_EXPORTS void read(const FileNode& ms, aruco::BoardConfiguration& m,
                        const aruco::BoardConfiguration& default_value = aruco::BoardConfiguration());

ARUCO_EXPORTS FileStorage& operator<<(FileStorage& fs, const aruco::Dictionary& m);
ARUCO_EXPORTS void read(const FileNode& ms, aruco::Dictionary& m,
                        const aruco::Dictionary& default_value = aruco::Dictionary());
}

#endif
