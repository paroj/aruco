/*
 * test_sample.cpp
 *
 *  Created on: 28.05.2015
 *      Author: parojtbe
 */

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "aruco.h"

#define TESTADATA_PATH "../testdata/"

static bool generateResults = false;

void storeMarkers(const std::vector<aruco::Marker>& markers, const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs << "Markers" << "[";

    for(auto& m : markers) {
        fs << "{";
        fs << "id" << m.id;
        fs << "Tvec" << m.Tvec;
        fs << "Rvec" << m.Rvec;
        fs << "}";
    }

    fs << "]";
}

std::vector<aruco::Marker> loadMarkers(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);

    std::vector<aruco::Marker> ret;

    for(const auto& ms : fs["Markers"]) {
        aruco::Marker m;
        ms["id"] >> m.id;
        ms["Tvec"] >> m.Tvec;
        ms["Rvec"] >> m.Rvec;
        ret.push_back(m);
    }

    return ret;
}


TEST(Aruco, Single) {
    using namespace aruco;

    aruco::CameraParameters CamParam;
    MarkerDetector MDetector;
    std::vector<Marker> Markers, expected;
    float MarkerSize = 1;

    // read the input image
    cv::Mat InImage = cv::imread(TESTADATA_PATH "single/image-test.png");

    // read camera parameters if specifed
    CamParam.readFromXMLFile(TESTADATA_PATH "single/intrinsics.yml");
    // resizes the parameters to fit the size of the input image
    CamParam.resize(InImage.size());

    // Ok, let's detect
    MDetector.detect(InImage, Markers, CamParam, MarkerSize);

    if(generateResults)
        storeMarkers(Markers, TESTADATA_PATH "single/expected.yml");

    expected = loadMarkers(TESTADATA_PATH "single/expected.yml");

    // now check the results
    EXPECT_EQ(expected.size(), Markers.size());

    for (auto i = 0; i < 6; i++) {
        EXPECT_EQ(expected[i].id, Markers[i].id);

        for(auto j = 0; j < 3; j++) {
            EXPECT_FLOAT_EQ(expected[i].Tvec[j], Markers[i].Tvec[j]);
            EXPECT_FLOAT_EQ(expected[i].Rvec[j], Markers[i].Rvec[j]);
        }
    }
}

TEST(Aruco, Board) {
    // TODO
}

TEST(Aruco, Chessboard) {
    // TODO use single frame from avi
}

TEST(Aruco, GL_Conversion) {
    // TODO check
    // TheCameraParams.glGetProjectionMatrix
    // TheMarkers[m].glGetModelViewMatrix
}

TEST(Aruco, HRM_CreateDictionary) {
    // TODO
}

TEST(Aruco, HRM_CreateBoard) {
    // TODO
}

TEST(Aruco, HRM_Single) {
    // use Testdata of Single
}

