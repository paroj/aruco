#include <gtest/gtest.h>

#include <aruco.h>
#include <chrono>
#include <opencv2/imgcodecs.hpp>

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

TEST( ArucoPerf, Single ){

    auto markerSize = 1.f;
    Mat testFrame;
    vector<Marker> detectedMarkers;
    MarkerDetector markerDetector;
    CameraParameters camParams;

    testFrame = imread( TESTDATA_PATH "single/image-test.png" );

    camParams.readFromXMLFile( TESTDATA_PATH "single/intrinsics.yml" );
    camParams.resize( testFrame.size()) ;

    auto startTime = steady_clock::now();

    for( auto i = 0; i < PERF_RUNS_DEFAULT; i++ )
        markerDetector.detect( testFrame, detectedMarkers, camParams, markerSize );

    auto avgProcessTime = ( (double) duration_cast<milliseconds>( steady_clock::now() - startTime ).count() / ( double) PERF_RUNS_DEFAULT );

    if( write_performance_data ){
        PerformanceData << "avg_marker_detection_time" << avgProcessTime;
        return;
    }

    double reference;
    PerformanceData["avg_marker_detection_time"] >> reference;
    EXPECT_LE(avgProcessTime, reference*TOLERANCE);
    BenchmarkData << "relative_marker_detection_speedup" << ( reference / avgProcessTime );
}

TEST( ArucoPerf, Board ){

    auto markerSize = 1.f;
    Mat testFrame;
    Board detectedBoard;
    vector<Marker> detectedMarkers;
    CameraParameters camParams;
    MarkerDetector markerDetector;
    BoardConfiguration boardConfig;
    BoardDetector boardDetector;

    testFrame = imread( TESTDATA_PATH "board/image-test.png" );

    boardConfig.readFromFile( TESTDATA_PATH "board/board_pix.yml" );

    camParams.readFromXMLFile( TESTDATA_PATH "board/intrinsics.yml" );
    camParams.resize( testFrame.size() );

    auto startTime = steady_clock::now();

    for( auto i = 0; i < PERF_RUNS_DEFAULT; i++ ){
        markerDetector.detect( testFrame, detectedMarkers );
        boardDetector.detect( detectedMarkers, boardConfig, detectedBoard, camParams, markerSize );
    }

    auto avgProcessTime = ( (double) duration_cast<milliseconds>( steady_clock::now() - startTime ).count() / (double) PERF_RUNS_DEFAULT );

    if( write_performance_data ){
        PerformanceData << "avg_board_detection_time" << avgProcessTime;
        return;
    }

    double reference;
    PerformanceData["avg_board_detection_time"] >> reference;
    EXPECT_LE(avgProcessTime, reference*TOLERANCE);
    BenchmarkData << "relative_board_detection_speedup" << ( reference / avgProcessTime );

}

TEST( ArucoPerf, Multi ) {

    auto markerSize = 1.f;
    double thresParam1, thresParam2;
    CameraParameters camParams;
    BoardConfiguration boardConfig;
    BoardDetector boardDetector;

    cv::Mat currentFrame = cv::imread(TESTDATA_PATH "chessboard/chessboard_frame.png");

    camParams.readFromXMLFile( TESTDATA_PATH "chessboard/intrinsics.yml" );
    camParams.resize( currentFrame.size() );

    boardConfig.readFromFile( TESTDATA_PATH "chessboard/chessboardinfo_pix.yml" );

    boardDetector.setParams( boardConfig, camParams, markerSize );
    boardDetector.getMarkerDetector().getThresholdParams( thresParam1, thresParam2 );
    boardDetector.getMarkerDetector().setCornerRefinementMethod( MarkerDetector::HARRIS );
    boardDetector.set_repj_err_thres( 1.5 );

    auto startTime = steady_clock::now();
    for (auto i = 0; i < PERF_RUNS_DEFAULT; i++) {
        boardDetector.detect(currentFrame);
    }

    auto thisTimeElapsed = duration_cast<milliseconds>(steady_clock::now() - startTime).count();
    double avgProcessTime = ((double)thisTimeElapsed / (double)PERF_RUNS_DEFAULT);

    if( write_performance_data ){
        PerformanceData << "avg_chessboard_detection_time" << avgProcessTime;
        return;
    }

    double reference;
    PerformanceData["avg_chessboard_detection_time"] >> reference;
    EXPECT_LE(avgProcessTime, reference*TOLERANCE);
    BenchmarkData << "relative_chessboard_detection_speedup" << ( reference / avgProcessTime );

}

TEST( ArucoPerf, GL_Conversion ) {

    float markerSize = 1.f;
    Mat testFrame;
    CameraParameters camParams;
    vector<Marker> detectedMarkers;
    Board detectedBoard;
    MarkerDetector markerDetector;
    BoardDetector boardDetector;
    BoardConfiguration boardConfig;

    testFrame = imread( TESTDATA_PATH "board/image-test.png" );
    boardConfig.readFromFile( TESTDATA_PATH "board/board_pix.yml" );

    camParams.readFromXMLFile( TESTDATA_PATH "board/intrinsics.yml" );
    camParams.resize( testFrame.size() );

    markerDetector.detect( testFrame, detectedMarkers, camParams, markerSize );
    boardDetector.detect( detectedMarkers, boardConfig, detectedBoard, camParams, markerSize );

    vector<Vec<double, 16>> gldata( detectedMarkers.size() + 2 );

    auto startTime = steady_clock::now();

    camParams.Distorsion.setTo(0); // silence cerr spam

    for ( auto run = 0; run < PERF_RUNS_DEFAULT; run++ ){
        camParams.glGetProjectionMatrix( testFrame.size(), testFrame.size(), gldata[0].val, 0.5, 10 );
        detectedBoard.glGetModelViewMatrix( gldata[1].val );
        for( size_t i = 0; i < detectedMarkers.size(); i++ )
            detectedMarkers[i].glGetModelViewMatrix( gldata[i + 2].val );
    }

    double avgProcessTime = ( (double) duration_cast<milliseconds>( steady_clock::now() - startTime ).count() / (double) PERF_RUNS_DEFAULT );

    if( write_performance_data ){
        PerformanceData << "avg_gl_conversion_time" << avgProcessTime;
        return;
    }

    double reference;
    PerformanceData["avg_gl_conversion_time"] >> reference;
    EXPECT_LE( avgProcessTime, reference*TOLERANCE);
    BenchmarkData << "relative_gl_conversion_speedup" << ( reference / avgProcessTime );
}

int main(int argc, char **argv) {

    ::testing::InitGoogleTest( &argc, argv );

    write_performance_data = not PerformanceData.open("/tmp/performance.yml", FileStorage::READ);

    if (write_performance_data) {
        PerformanceData.open("/tmp/performance.yml", FileStorage::WRITE);
        PerformanceData << "aruco_version" << ARUCO_VERSION;
        PerformanceData << "opencv_version" << OPENCV_VERSION;
    } else {
        BenchmarkData.open("/tmp/benchmark.yml", FileStorage::WRITE);
        BenchmarkData["aruco_version"] >> Aruco_Version;
        BenchmarkData["opencv_version"] >> Opencv_Version;
    }

    return RUN_ALL_TESTS();
}
