#include <gtest/gtest.h>

#include <aruco.h>
#include <chrono>
#include <opencv2/highgui/highgui.hpp>

#ifndef TEST_MODE
    #define PERFORMANCE_TEST false
    #define BENCHMARK_TEST true
#else
    #define PERFORMANCE_TEST true
    #define BENCHMARK_TEST false
#endif

#define PERF_RUNS_MULTI 5
#define PERF_RUNS_DEFAULT 1000

#define TESTDATA_PATH "../testdata/"

using namespace cv;
using namespace aruco;
using namespace std;
using namespace std::chrono;

static string Aruco_Version;
static string Opencv_Version;

static FileStorage BenchmarkData;
static FileStorage PerformanceData;

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

    if( PERFORMANCE_TEST ){
        PerformanceData << "avg_marker_detection_time" << avgProcessTime;
        return;
    }

    EXPECT_LT( (double) PerformanceData["avg_marker_detection_time"], avgProcessTime );
    BenchmarkData << "relative_marker_detection_speedup" << ( (double) PerformanceData["avg_marker_detection_time"] / (double) avgProcessTime );
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

    if( PERFORMANCE_TEST ){
        PerformanceData << "avg_board_detection_time" << avgProcessTime;
        return;
    }

    EXPECT_LT( (double) PerformanceData["avg_board_detection_time"], avgProcessTime );
    BenchmarkData << "relative_board_detection_speedup" << ( (double) PerformanceData["avg_board_detection_time"] / (double) avgProcessTime );

}

TEST( ArucoPerf, Multi ) {

    auto markerSize = 1.f;
    double thresParam1, thresParam2;
    Mat currentFrame;
    VideoCapture video;
    CameraParameters camParams;
    BoardConfiguration boardConfig;
    BoardDetector boardDetector;

    if ( !video.open( TESTDATA_PATH "chessboard/chessboard.mp4" ) )
        FAIL();

    video >> currentFrame;

    camParams.readFromXMLFile( TESTDATA_PATH "chessboard/intrinsics.yml" );
    camParams.resize( currentFrame.size() );

    boardConfig.readFromFile( TESTDATA_PATH "chessboard/chessboardinfo_pix.yml" );

    boardDetector.setParams( boardConfig, camParams, markerSize );
    boardDetector.getMarkerDetector().getThresholdParams( thresParam1, thresParam2 );
    boardDetector.getMarkerDetector().setCornerRefinementMethod( MarkerDetector::HARRIS );
    boardDetector.set_repj_err_thres( 1.5 );

    auto run = 0;
    auto videoPosition = 0;
    auto thisTimeElapsed = 0;

    do {

        video.retrieve( currentFrame );

        auto startTime = steady_clock::now();
        boardDetector.detect( currentFrame );
        thisTimeElapsed += duration_cast<milliseconds>( steady_clock::now() - startTime ).count();

        videoPosition++;
        if (  videoPosition == (int) video.get( CV_CAP_PROP_FRAME_COUNT ) ){
            videoPosition = 0;
            video.set( CV_CAP_PROP_POS_FRAMES, videoPosition );
            run++;
        }

    } while ( video.grab() && run < PERF_RUNS_MULTI );

    double avgProcessTime = ( (double) thisTimeElapsed / (double) PERF_RUNS_MULTI );

    if( PERFORMANCE_TEST ){
        PerformanceData << "avg_chessboard_detection_time" << avgProcessTime;
        return;
    }

    EXPECT_LT( (double) PerformanceData["avg_chessboard_detection_time"], avgProcessTime );
    BenchmarkData << "relative_chessboard_detection_speedup" << ( (double) PerformanceData["avg_chessboard_detection_time"] / (double) avgProcessTime );

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

    for ( auto run = 0; run < PERF_RUNS_DEFAULT; run++ ){
        camParams.glGetProjectionMatrix( testFrame.size(), testFrame.size(), gldata[0].val, 0.5, 10 );
        detectedBoard.glGetModelViewMatrix( gldata[1].val );
        for( auto i = 0; i < detectedMarkers.size(); i++ )
            detectedMarkers[i].glGetModelViewMatrix( gldata[i + 2].val );
    }

    double avgProcessTime = ( (double) duration_cast<milliseconds>( steady_clock::now() - startTime ).count() / (double) PERF_RUNS_DEFAULT );

    if( PERFORMANCE_TEST ){
        PerformanceData << "avg_gl_conversion_time" << avgProcessTime;
        return;
    }

    EXPECT_LT( (double) PerformanceData["avg_gl_conversion_time"], avgProcessTime );
    BenchmarkData << "relative_gl_conversion_speedup" << ( (double) PerformanceData["avg_gl_conversion_time"] / (double) avgProcessTime );

}

int main(int argc, char **argv) {

    ::testing::InitGoogleTest( &argc, argv );

    if ( PERFORMANCE_TEST ){
        PerformanceData.open( TESTDATA_PATH "performance.yml", 1 );
        BenchmarkData.open( TESTDATA_PATH "benchmark.yml", 0 );
        PerformanceData << "aruco_version" << ARUCO_VERSION;
        PerformanceData << "opencv_version" << OPENCV_VERSION;
    }
    else if ( BENCHMARK_TEST ){
        PerformanceData.open( TESTDATA_PATH "performance.yml", 0 );
        BenchmarkData.open( TESTDATA_PATH "benchmark.yml", 1 );
        BenchmarkData["aruco_version"] >> Aruco_Version;
        BenchmarkData["opencv_version"] >> Opencv_Version;
    }
    bool res = RUN_ALL_TESTS();

    PerformanceData.release();
    BenchmarkData.release();

    return res;
}
