/*
 * test_sample.cpp
 *
 *  Created on: 28.05.2015
 *      Author: parojtbe
 */

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include <aruco.h>
#include "filestorage_adapter.h"

#define TESTADATA_PATH "../testdata/"

static bool generateResults = false;

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

    int mode = cv::FileStorage::Mode(generateResults);
    cv::FileStorage fs(TESTADATA_PATH "single/expected.yml", mode);

    if(generateResults) {
        fs << "Markers" << Markers;
        return;
    }

    fs["Markers"] >> expected;

    // now check the results
    EXPECT_EQ(expected.size(), Markers.size());

    for (size_t i = 0; i < Markers.size(); i++) {
        EXPECT_EQ(expected[i].id, Markers[i].id);

        EXPECT_FLOAT_EQ(expected[i].getCenter().x, Markers[i].getCenter().x);
        EXPECT_FLOAT_EQ(expected[i].getCenter().y, Markers[i].getCenter().y);

        for(auto j = 0; j < 3; j++) {
            EXPECT_FLOAT_EQ(expected[i].Tvec[j], Markers[i].Tvec[j]);
            EXPECT_FLOAT_EQ(expected[i].Rvec[j], Markers[i].Rvec[j]);
        }
    }
}

TEST(Aruco, Board) {
    using namespace aruco;
    aruco::CameraParameters CamParam;
    MarkerDetector MDetector;
    vector<Marker> Markers;
    float MarkerSize = 1;
    BoardConfiguration TheBoardConfig;
    BoardDetector TheBoardDetector;
    Board TheBoardDetected, expected;

    cv::Mat InImage = cv::imread(TESTADATA_PATH "board/image-test.png");
    TheBoardConfig.readFromFile(TESTADATA_PATH "board/board_pix.yml");

    CamParam.readFromXMLFile(TESTADATA_PATH "board/intrinsics.yml");
    // resizes the parameters to fit the size of the input image
    CamParam.resize(InImage.size());

    MDetector.detect(InImage, Markers); // detect markers without computing R and T information
    // Detection of the board
    TheBoardDetector.detect(Markers, TheBoardConfig, TheBoardDetected, CamParam, MarkerSize);

    int mode = cv::FileStorage::Mode(generateResults);
    cv::FileStorage fs(TESTADATA_PATH "board/expected.yml", mode);

    if(generateResults) {
        fs << "Board" << TheBoardDetected;
        return;
    }

    fs["Board"] >> expected;

    // now check the results
    EXPECT_EQ(expected.size(), TheBoardDetected.size());
    for(int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(expected.Rvec.at<float>(i), TheBoardDetected.Rvec.at<float>(i));
        EXPECT_FLOAT_EQ(expected.Tvec.at<float>(i), TheBoardDetected.Tvec.at<float>(i));
    }
}

TEST(Aruco, Chessboard) {
    // TODO use single frame from avi
}

TEST(Aruco, GL_Conversion) {
    using namespace aruco;
    aruco::CameraParameters CamParam;
    MarkerDetector MDetector;
    vector<Marker> Markers;
    float MarkerSize = 1;
    BoardConfiguration TheBoardConfig;
    BoardDetector TheBoardDetector;
    Board TheBoardDetected;

    cv::Mat InImage = cv::imread(TESTADATA_PATH "board/image-test.png");
    TheBoardConfig.readFromFile(TESTADATA_PATH "board/board_pix.yml");

    CamParam.readFromXMLFile(TESTADATA_PATH "board/intrinsics.yml");
    // resizes the parameters to fit the size of the input image
    CamParam.resize(InImage.size());

    MDetector.detect(InImage, Markers, CamParam, MarkerSize); // detect markers computing R and T information
    // Detection of the board
    TheBoardDetector.detect(Markers, TheBoardConfig, TheBoardDetected, CamParam, MarkerSize);

    int mode = cv::FileStorage::Mode(generateResults);
    cv::FileStorage fs(TESTADATA_PATH "board/expected_gl.yml", mode);

    std::vector<cv::Vec<double, 16>> gldata(Markers.size() + 2), expected;

    CamParam.glGetProjectionMatrix(InImage.size(), InImage.size(), gldata[0].val, 0.5, 10);
    TheBoardDetected.glGetModelViewMatrix(gldata[1].val);

    for(size_t i = 0; i < Markers.size(); i++) {
        Markers[i].glGetModelViewMatrix(gldata[i + 2].val);
    }

    if(generateResults) {
        fs << "gldata" << "[";
        for(size_t i = 0; i < gldata.size(); i++) {
            fs << gldata[i];
        }
        fs << "]";
        return;
    }

    // load results
    cv::FileNode gls = fs["gldata"];
    expected.resize(gls.size());

    for(size_t i = 0; i < expected.size(); i++) {
        gls[i] >> expected[i];
    }

    // now check the results
    for(size_t i = 0; i < expected.size(); i++) {
        for(int j = 0; j < 16; j++) {
            EXPECT_FLOAT_EQ(expected[i].val[j], gldata[i].val[j]);
        }
    }
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

TEST(ArucoPerf, Single) {
    // ATTENTION: finish normal tests first
    // TODO something like (attention pseudocode):
    // auto t0 = std::chrono::steady_clock::now();
    for(int i = 0; i < 1000; i++) {
        // MDetector.detect(InImage, Markers, CamParam, MarkerSize);
    }
    // auto t = std::chrono::steady_clock::now() - t0;
    // auto first_run = -1;
    // if(/var/tmp/something.xx exists)
    //     first_run = read t from /var/tmp/something.xx
    // if(first_run > 0) {
    //    EXPECT_LT(t, first_run);
    // } else {
    //    write t to file in /var/tmp/something.xx
    // }
}
