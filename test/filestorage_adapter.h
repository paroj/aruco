/*
 * filestorage_adapter.h
 *
 *  Created on: 26.06.2015
 *      Author: parojtbe
 */

#pragma once

#include "aruco.h"

namespace cv {
inline FileStorage& operator<<(FileStorage& fs, const std::vector<aruco::Marker>& markers) {
    fs << "[";
    for(auto& m : markers) {
        fs << m;
    }
    fs << "]";

    return fs;
}

inline FileStorage& operator<<(FileStorage& fs, const aruco::Board& b) {
    fs << "{";
    fs << "Tvec" << Vec3d(b.Tvec);
    fs << "Rvec" << Vec3d(b.Rvec);
    fs << "Markers" << "[";

    for(auto& m : b) {
        fs << m;
    }

    fs << "]";
    fs << "}";

    return fs;
}

inline void read(const FileNode& bs, aruco::Board& b, const aruco::Board& default_value = aruco::Board()) {
    if(bs.empty()) {
        b = default_value;
        return;
    }

    Vec3d Tvec;
    Vec3d Rvec;

    bs["Tvec"] >> Tvec;
    bs["Rvec"] >> Rvec;

    b.Rvec = Mat(Rvec);
    b.Tvec = Mat(Tvec);

    const FileNode& markers = bs["Markers"];
    b.resize(markers.size());

    for(size_t i = 0; i < b.size(); i++) {
        markers[i] >> b[i];
    }
}
}
