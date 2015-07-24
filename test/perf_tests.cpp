#include <gtest/gtest.h>

#include <aruco.h>
#include "highlyreliablemarkers.h"

#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/version.hpp>

#include "test.h"

#define PERF_RUNS_DEFAULT 1000
#define TESTDATA_PATH "../testdata/"
#define TOLERANCE 1.05

using namespace cv;
using namespace aruco;
using namespace std;
using namespace std::chrono;

namespace {
bool write_performance_data;
string Aruco_Version;
string Opencv_Version;

FileStorage BenchmarkData;
FileStorage PerformanceData;
}

TEST(ArucoPerf, Single) {
    MarkerFixture mf;
    // color conversion should not be part of performance test
    Mat testFrame = imread(TESTDATA_PATH "single/image-test.png", IMREAD_GRAYSCALE);

    mf.CamParam.readFromXMLFile(TESTDATA_PATH "single/intrinsics.yml");
    mf.CamParam.resize(testFrame.size());

    auto startTime = steady_clock::now();

    for (auto i = 0; i < PERF_RUNS_DEFAULT; i++)
        mf.MDetector.detect(testFrame, mf.Markers, mf.CamParam, mf.MarkerSize);

    auto avgProcessTime = ((double)duration_cast<milliseconds>(steady_clock::now() - startTime).count() /
                           (double)PERF_RUNS_DEFAULT);

    if (write_performance_data) {
        PerformanceData << "avg_marker_detection_time" << avgProcessTime;
        return;
    }

    double reference;
    PerformanceData["avg_marker_detection_time"] >> reference;
    EXPECT_LE(avgProcessTime, reference * TOLERANCE);
    BenchmarkData << "relative_marker_detection_speedup" << (reference / avgProcessTime);
}

TEST(ArucoPerf, Board) {
    MarkerBoardFixture mbf;
    Mat testFrame = imread(TESTDATA_PATH "board/image-test.png", IMREAD_GRAYSCALE);

    mbf.BoardConfig.readFromFile(TESTDATA_PATH "board/board_pix.yml");

    mbf.CamParam.readFromXMLFile(TESTDATA_PATH "board/intrinsics.yml");
    mbf.CamParam.resize(testFrame.size());

    auto startTime = steady_clock::now();

    for (auto i = 0; i < PERF_RUNS_DEFAULT; i++) {
        mbf.MDetector.detect(testFrame, mbf.Markers);
        mbf.BoardDetector.detect(mbf.Markers, mbf.BoardConfig, mbf.Board, mbf.CamParam, mbf.MarkerSize);
    }

    auto avgProcessTime = ((double)duration_cast<milliseconds>(steady_clock::now() - startTime).count() /
                           (double)PERF_RUNS_DEFAULT);

    if (write_performance_data) {
        PerformanceData << "avg_board_detection_time" << avgProcessTime;
        return;
    }

    double reference;
    PerformanceData["avg_board_detection_time"] >> reference;
    EXPECT_LE(avgProcessTime, reference * TOLERANCE);
    BenchmarkData << "relative_board_detection_speedup" << (reference / avgProcessTime);
}

TEST(ArucoPerf, Multi) {
    MarkerBoardFixture mbf;

    cv::Mat currentFrame = cv::imread(TESTDATA_PATH "chessboard/chessboard_frame.png", IMREAD_GRAYSCALE);

    mbf.CamParam.readFromXMLFile(TESTDATA_PATH "chessboard/intrinsics.yml");
    mbf.CamParam.resize(currentFrame.size());

    mbf.BoardConfig.readFromFile(TESTDATA_PATH "chessboard/chessboardinfo_pix.yml");

    mbf.BoardDetector.setParams(mbf.BoardConfig, mbf.CamParam, mbf.MarkerSize);
    mbf.BoardDetector.getMarkerDetector().setCornerRefinementMethod(MarkerDetector::HARRIS);
    mbf.BoardDetector.set_repj_err_thres(1.5);

    auto startTime = steady_clock::now();
    for (auto i = 0; i < PERF_RUNS_DEFAULT; i++) {
        mbf.BoardDetector.detect(currentFrame);
    }

    auto thisTimeElapsed = duration_cast<milliseconds>(steady_clock::now() - startTime).count();
    double avgProcessTime = ((double)thisTimeElapsed / (double)PERF_RUNS_DEFAULT);

    if (write_performance_data) {
        PerformanceData << "avg_chessboard_detection_time" << avgProcessTime;
        return;
    }

    double reference;
    PerformanceData["avg_chessboard_detection_time"] >> reference;
    EXPECT_LE(avgProcessTime, reference * TOLERANCE);
    BenchmarkData << "relative_chessboard_detection_speedup" << (reference / avgProcessTime);
}

TEST(ArucoPerf, GL_Conversion) {
    MarkerBoardFixture mbf;
    Mat testFrame = imread(TESTDATA_PATH "board/image-test.png");

    mbf.BoardConfig.readFromFile(TESTDATA_PATH "board/board_pix.yml");

    mbf.CamParam.readFromXMLFile(TESTDATA_PATH "board/intrinsics.yml");
    mbf.CamParam.resize(testFrame.size());

    mbf.MDetector.detect(testFrame, mbf.Markers, mbf.CamParam, mbf.MarkerSize);
    mbf.BoardDetector.detect(mbf.Markers, mbf.BoardConfig, mbf.Board, mbf.CamParam, mbf.MarkerSize);

    vector<Vec<double, 16> > gldata(mbf.Markers.size() + 2);

    auto startTime = steady_clock::now();

    mbf.CamParam.Distorsion.setTo(0); // silence cerr spam

    for (auto run = 0; run < PERF_RUNS_DEFAULT; run++) {
        mbf.CamParam.glGetProjectionMatrix(testFrame.size(), testFrame.size(), gldata[0].val, 0.5, 10);
        mbf.Board.glGetModelViewMatrix(gldata[1].val);
        for (size_t i = 0; i < mbf.Markers.size(); i++)
            mbf.Markers[i].glGetModelViewMatrix(gldata[i + 2].val);
    }

    double avgProcessTime = ((double)duration_cast<milliseconds>(steady_clock::now() - startTime).count() /
                             (double)PERF_RUNS_DEFAULT);

    if (write_performance_data) {
        PerformanceData << "avg_gl_conversion_time" << avgProcessTime;
        return;
    }

    double reference;
    PerformanceData["avg_gl_conversion_time"] >> reference;
    EXPECT_LE(avgProcessTime, reference * TOLERANCE);
    BenchmarkData << "relative_gl_conversion_speedup" << (reference / avgProcessTime);
}

TEST(ArucoPerf, HRM_Single) {
    MarkerFixture mf;

    Dictionary dictionary;
    dictionary.fromFile(TESTDATA_PATH "hrm/dictionaries/d4x4_100.yml");
    HighlyReliableMarkers::loadDictionary(dictionary);

    Mat frameImage = imread(TESTDATA_PATH "hrm/image-test.png", IMREAD_GRAYSCALE);

    mf.CamParam.readFromXMLFile(TESTDATA_PATH "hrm/intrinsics.yml");
    mf.CamParam.resize(frameImage.size());

    mf.MDetector.enableLockedCornersMethod(false);
    mf.MDetector.setMakerDetectorFunction(HighlyReliableMarkers::detect);
    mf.MDetector.setThresholdParams(21, 7);
    mf.MDetector.setCornerRefinementMethod(MarkerDetector::LINES);
    mf.MDetector.setWarpSize((dictionary[0].n() + 2) * 8);
    mf.MDetector.setMinMaxSize(0.005, 0.5);

    auto startTime = steady_clock::now();

    for (auto run = 0; run < PERF_RUNS_DEFAULT; run++) {
        mf.MDetector.detect(frameImage, mf.Markers, mf.CamParam, mf.MarkerSize);
    }

    double avgProcessTime = ((double)duration_cast<milliseconds>(steady_clock::now() - startTime).count() /
                             (double)PERF_RUNS_DEFAULT);

    if (write_performance_data) {
        PerformanceData << "avg_hrm_marker_detection_time" << avgProcessTime;
        return;
    }

    double reference;
    PerformanceData["avg_hrm_marker_detection_time"] >> reference;
    EXPECT_LE(avgProcessTime, reference * TOLERANCE);
    BenchmarkData << "relative_hrm_marker_detection_speedup" << (reference / avgProcessTime);
}

int main(int argc, char** argv) {

    ::testing::InitGoogleTest(&argc, argv);

    write_performance_data = not PerformanceData.open("/tmp/performance.yml", FileStorage::READ);

    if (write_performance_data) {
        PerformanceData.open("/tmp/performance.yml", FileStorage::WRITE);
        PerformanceData << "aruco_version" << ARUCO_VERSION;
        PerformanceData << "opencv_version" << CV_VERSION;
    } else {
        BenchmarkData.open("/tmp/benchmark.yml", FileStorage::WRITE);
        BenchmarkData["aruco_version"] >> Aruco_Version;
        BenchmarkData["opencv_version"] >> Opencv_Version;
    }

    return RUN_ALL_TESTS();
}
