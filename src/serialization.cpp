/*
 * serialization.cpp
 *
 *  Created on: 01.09.2015
 *      Author: parojtbe
 */

#include "serialization.h"

#include <sstream>
#include <string>

namespace {
// convert to string
template <class T> static std::string to_string(T num) {
    std::stringstream ss;
    ss << num;
    return ss.str();
}
}

namespace cv {
// Marker
FileStorage& operator<<(FileStorage& fs, const aruco::Marker& m) {
    fs << "{";
    fs << "id" << m.id;

    if (!m.Tvec.empty() && !m.Rvec.empty()) {
        fs << "Tvec" << Vec3d(m.Tvec);
        fs << "Rvec" << Vec3d(m.Rvec);
    }

    fs << "corners"
       << "[:";

    for (size_t i = 0; i < m.size(); i++) {
        const Point2f& c = m[i];
        fs << c;
    }

    fs << "]";
    fs << "}";

    return fs;
}

void read(const FileNode& ms, aruco::Marker& m, const aruco::Marker& default_value) {
    if (ms.empty()) {
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

    for (size_t i = 0; i < m.size(); i++) {
        corners[i] >> m[i];
    }
}

// BoardConfiguration
FileStorage& operator<<(FileStorage& fs, const aruco::BoardConfiguration& m) {
    fs << "aruco_bc_nmarkers" << (int)m.size();
    fs << "aruco_bc_mInfoType" << (int)m.mInfoType;
    fs << "aruco_bc_markers"
       << "[";
    for (size_t i = 0; i < m.size(); i++) {
        fs << "{:"
           << "id" << m[i].id;

        fs << "corners"
           << "[:";
        for (int c = 0; c < 4; c++)
            fs << m[i][c];
        fs << "]";
        fs << "}";
    }
    fs << "]";

    return fs;
}

void read(const FileNode& fn, aruco::BoardConfiguration& bc,
          const aruco::BoardConfiguration& default_value) {
    // look for the nmarkers
    if (fn["aruco_bc_nmarkers"].empty())
        CV_Error(cv::Error::StsUnsupportedFormat, "invalid file type");

    int bc_nmarkers;
    fn["aruco_bc_nmarkers"] >> bc_nmarkers;
    fn["aruco_bc_mInfoType"] >> bc.mInfoType;
    cv::FileNode markers = fn["aruco_bc_markers"];

    CV_Assert(bc_nmarkers == int(markers.size()));

    bc.resize(markers.size());
    int i = 0;
    for (FileNodeIterator it = markers.begin(); it != markers.end(); ++it, i++) {
        bc[i].id = (*it)["id"];
        const FileNode& FnCorners = (*it)["corners"];
        CV_Assert(FnCorners.size() == 4);

        bc[i].resize(4);
        for (size_t c = 0; c < 4; c++) {
            FnCorners[c] >> bc[i][c];
        }
    }
}

// Dictionary
FileStorage& operator<<(FileStorage& fs, const aruco::Dictionary& d) {
    fs << "nmarkers" << (int)d.size();             // cardinal of D
    fs << "markersize" << int(d[0].n()); // n
    fs << "tau0" << d.tau0;           // n
    // save each marker code
    for (size_t i = 0; i < d.size(); i++) {
        fs << "marker_" + to_string(i) << d[i].toString();
    }
    return fs;
}

void read(const FileNode& fs, aruco::Dictionary& d,
          const aruco::Dictionary& default_value) {
    int nmarkers, markersize;

    // read number of markers
    fs["nmarkers"] >> nmarkers;     // cardinal of D
    fs["markersize"] >> markersize; // n
    fs["tau0"] >> d.tau0;

    // read each marker info
    for (int i = 0; i < nmarkers; i++) {
        std::string s;
        fs["marker_" + to_string(i)] >> s;
        aruco::MarkerCode m(markersize);
        m.fromString(s);
        d.push_back(m);
    }
}
}
