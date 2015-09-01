/*
 * filestorage_adapter.h
 *
 *  Created on: 26.06.2015
 *      Author: parojtbe
 */

#pragma once

#include "aruco.h"

namespace cv {
inline FileStorage& operator<<(FileStorage& fs, const aruco::Marker& m) {
    fs << "{";
    fs << "id" << m.id;

    if(!m.Tvec.empty() && !m.Rvec.empty()) {
        fs << "Tvec" << Vec3d(m.Tvec);
        fs << "Rvec" << Vec3d(m.Rvec);
    }

    fs << "corners" << "[:";

    for(auto& c : m) {
        fs << c;
    }

    fs << "]";
    fs << "}";

    return fs;
}

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

inline void read(const FileNode& ms, aruco::Marker& m, const aruco::Marker& default_value = aruco::Marker()) {
    if(ms.empty()) {
        m = default_value;
        return;
    }

    Vec3d Tvec;
    Vec3d Rvec;

    ms["id"] >> m.id;

    ms["Tvec"] >> Tvec;
    ms["Rvec"] >> Rvec;

    m.Rvec = Mat(Rvec);
    m.Tvec = Mat(Tvec);

    const FileNode& corners = ms["corners"];
    m.resize(corners.size());

    for(size_t i = 0; i < m.size(); i++) {
        corners[i] >> m[i];
    }
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
