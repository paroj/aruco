#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <aruco.h>
#include "arucofidmarkers.h"
#include "filestorage_adapter.h"
#include "highlyreliablemarkers.h"
#include "test.h"

#define TESTDATA_PATH "../testdata/"

namespace {
bool generateResults = false;

void compareBoardConfig(aruco::BoardConfiguration expectedBoard, aruco::BoardConfiguration currentBoard){
    ASSERT_EQ( expectedBoard.size(), currentBoard.size() );

    for (size_t i = 0; i < currentBoard.size(); i++){
        EXPECT_EQ(expectedBoard[i].id, currentBoard[i].id);
        for (int j = 0; j < 4; j++){
            EXPECT_FLOAT_EQ(expectedBoard[i].at(j).x, currentBoard[i].at(j).x);
            EXPECT_FLOAT_EQ(expectedBoard[i].at(j).y, currentBoard[i].at(j).y);
            EXPECT_FLOAT_EQ(expectedBoard[i].at(j).z, currentBoard[i].at(j).z);
        }
    }
}

}

TEST(Aruco, CreateMarker){

    const int pixSize = 500;
    const int markerId = 471;

    if( generateResults ){
        cv::Mat marker = aruco::FiducidalMarkers::createMarkerImage(markerId, pixSize, true, true);
        cv::imwrite(TESTDATA_PATH "board/locked-watermark-marker-expected.png", marker);

        marker = aruco::FiducidalMarkers::createMarkerImage(markerId, pixSize, false, true);
        cv::imwrite(TESTDATA_PATH "board/locked-marker-expected.png", marker);

        marker = aruco::FiducidalMarkers::createMarkerImage(markerId, pixSize, false, false);
        cv::imwrite(TESTDATA_PATH "board/marker-expected.png", marker);

        marker = aruco::FiducidalMarkers::createMarkerImage(markerId, pixSize, true, false);
        cv::imwrite(TESTDATA_PATH "board/wartermark-marker-expected.png", marker);

        return ;
    }

    cv::Mat diff, expectedMarker, currentMarker;

    expectedMarker = cv::imread(TESTDATA_PATH "board/locked-watermark-marker-expected.png", cv::IMREAD_GRAYSCALE);
    currentMarker = aruco::FiducidalMarkers::createMarkerImage(markerId, pixSize, true, true);
    cv::compare(expectedMarker, currentMarker, diff, cv::CMP_NE);
    EXPECT_EQ(0, cv::countNonZero(diff));

    expectedMarker = cv::imread(TESTDATA_PATH "board/locked-marker-expected.png", cv::IMREAD_GRAYSCALE);
    currentMarker = aruco::FiducidalMarkers::createMarkerImage(markerId, pixSize, false, true);
    cv::compare(expectedMarker, currentMarker, diff, cv::CMP_NE);
    EXPECT_EQ(0, cv::countNonZero(diff));

    expectedMarker = cv::imread(TESTDATA_PATH "board/marker-expected.png", cv::IMREAD_GRAYSCALE);
    currentMarker = aruco::FiducidalMarkers::createMarkerImage(markerId, pixSize, false, false);
    cv::compare(expectedMarker, currentMarker, diff, cv::CMP_NE);
    EXPECT_EQ(0, cv::countNonZero(diff));

    expectedMarker = cv::imread(TESTDATA_PATH "board/wartermark-marker-expected.png", cv::IMREAD_GRAYSCALE);
    currentMarker = aruco::FiducidalMarkers::createMarkerImage(markerId, pixSize, true, false);
    cv::compare(expectedMarker, currentMarker, diff, cv::CMP_NE);
    EXPECT_EQ(0, cv::countNonZero(diff));

}

TEST(Aruco, Single) {
    MarkerFixture mf;
    std::vector<aruco::Marker> expected;

    // read the input image
    cv::Mat InImage = cv::imread(TESTDATA_PATH "single/image-test.png");

    // read camera parameters if specifed
    mf.CamParam.readFromXMLFile(TESTDATA_PATH "single/intrinsics.yml");
    // resizes the parameters to fit the size of the input image
    mf.CamParam.resize(InImage.size());

    // Ok, let's detect
    mf.MDetector.detect(InImage, mf.Markers, mf.CamParam, mf.MarkerSize);

    int mode = cv::FileStorage::Mode(generateResults);
    cv::FileStorage fs(TESTDATA_PATH "single/expected.yml", mode);

    if (generateResults) {
        fs << "Markers" << mf.Markers;
        return;
    }

    fs["Markers"] >> expected;

    // now check the results
    ASSERT_EQ(expected.size(), mf.Markers.size());

    for (size_t i = 0; i < mf.Markers.size(); i++) {
        EXPECT_EQ(expected[i].id, mf.Markers[i].id);

        EXPECT_FLOAT_EQ(expected[i].getCenter().x, mf.Markers[i].getCenter().x);
        EXPECT_FLOAT_EQ(expected[i].getCenter().y, mf.Markers[i].getCenter().y);

        for (auto j = 0; j < 3; j++) {
            EXPECT_FLOAT_EQ(expected[i].Tvec(j), mf.Markers[i].Tvec(j));
            EXPECT_FLOAT_EQ(expected[i].Rvec(j), mf.Markers[i].Rvec(j));
        }
    }
}

TEST(Aruco, CreateBoard){

    const float interMarkerDistance = 0.2;
    const int XSize = 5,
              YSize = 5,
              pixSize = 100;

    cv::theRNG().state = 4711;

    if ( generateResults ){

        aruco::BoardConfiguration DefaultBoard, ChessBoard, FrameBord;

        aruco::FiducidalMarkers::createBoardImage(cv::Size(XSize, YSize), pixSize, pixSize * interMarkerDistance, DefaultBoard);
        aruco::FiducidalMarkers::createBoardImage_ChessBoard(cv::Size(XSize, YSize), pixSize, ChessBoard);
        aruco::FiducidalMarkers::createBoardImage_Frame(cv::Size(XSize, YSize), pixSize, pixSize * interMarkerDistance, FrameBord);

        DefaultBoard.saveToFile(TESTDATA_PATH "board/defaultBoard-expected.yml");
        ChessBoard.saveToFile(TESTDATA_PATH "board/chessBoard-expected.yml");
        FrameBord.saveToFile(TESTDATA_PATH "board/frameBoard-expected.yml");

        return ;
    }

    aruco::BoardConfiguration ExpectedBoard, CurrentBoard;
    ExpectedBoard.readFromFile(TESTDATA_PATH "board/defaultBoard-expected.yml");
    aruco::FiducidalMarkers::createBoardImage(cv::Size(XSize, YSize), pixSize, pixSize * interMarkerDistance, CurrentBoard);
    compareBoardConfig( ExpectedBoard, CurrentBoard );

    ExpectedBoard.clear(); CurrentBoard.clear();
    ExpectedBoard.readFromFile(TESTDATA_PATH "board/chessBoard-expected.yml");
    aruco::FiducidalMarkers::createBoardImage_ChessBoard(cv::Size(XSize, YSize), pixSize, CurrentBoard);
    compareBoardConfig( ExpectedBoard, CurrentBoard );

    ExpectedBoard.clear(); CurrentBoard.clear();
    ExpectedBoard.readFromFile(TESTDATA_PATH "board/frameBoard-expected.yml");
    aruco::FiducidalMarkers::createBoardImage_Frame(cv::Size(XSize, YSize), pixSize, pixSize * interMarkerDistance, CurrentBoard);
    compareBoardConfig( ExpectedBoard, CurrentBoard );

}

TEST(Aruco, Board) {
    MarkerBoardFixture mbf;
    aruco::Board expected;

    cv::Mat InImage = cv::imread(TESTDATA_PATH "board/image-test.png");
    mbf.BoardConfig.readFromFile(TESTDATA_PATH "board/board_pix.yml");

    mbf.CamParam.readFromXMLFile(TESTDATA_PATH "board/intrinsics.yml");
    // resizes the parameters to fit the size of the input image
    mbf.CamParam.resize(InImage.size());

    mbf.MDetector.detect(InImage, mbf.Markers); // detect markers without computing R and T information
    // Detection of the board
    mbf.BoardDetector.detect(mbf.Markers, mbf.BoardConfig, mbf.Board, mbf.CamParam, mbf.MarkerSize);

    int mode = cv::FileStorage::Mode(generateResults);
    cv::FileStorage fs(TESTDATA_PATH "board/expected.yml", mode);

    if (generateResults) {
        fs << "Board" << mbf.Board;
        return;
    }

    fs["Board"] >> expected;

    // now check the results
    EXPECT_EQ(expected.size(), mbf.Board.size());
    for (int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(expected.Rvec(i), mbf.Board.Rvec(i));
        EXPECT_FLOAT_EQ(expected.Tvec(i), mbf.Board.Tvec(i));
    }
}

TEST(Aruco, Multi) {
    MarkerBoardFixture mbf;
    aruco::Board expected;

    cv::Mat InImage = cv::imread(TESTDATA_PATH "chessboard/chessboard_frame.png");
    mbf.BoardConfig.readFromFile(TESTDATA_PATH "chessboard/chessboardinfo_pix.yml");

    mbf.CamParam.readFromXMLFile(TESTDATA_PATH "chessboard/intrinsics.yml");
    // resizes the parameters to fit the size of the input image
    mbf.CamParam.resize(InImage.size());

    mbf.MDetector.detect(InImage, mbf.Markers); // detect markers without computing R and T information
    // Detection of the board
    mbf.BoardDetector.detect(mbf.Markers, mbf.BoardConfig, mbf.Board, mbf.CamParam, mbf.MarkerSize);

    int mode = cv::FileStorage::Mode(generateResults);
    cv::FileStorage fs(TESTDATA_PATH "chessboard/expected.yml", mode);

    if (generateResults) {
        fs << "Board" << mbf.Board;
        return;
    }

    fs["Board"] >> expected;

    // now check the results
    EXPECT_EQ(expected.size(), mbf.Board.size());
    for (int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(expected.Rvec(i), mbf.Board.Rvec(i));
        EXPECT_FLOAT_EQ(expected.Tvec(i), mbf.Board.Tvec(i));
    }
}

TEST(Aruco, GL_Conversion) {
    MarkerBoardFixture mbf;

    cv::Mat InImage = cv::imread(TESTDATA_PATH "board/image-test.png");
    mbf.BoardConfig.readFromFile(TESTDATA_PATH "board/board_pix.yml");

    mbf.CamParam.readFromXMLFile(TESTDATA_PATH "board/intrinsics.yml");
    // resizes the parameters to fit the size of the input image
    mbf.CamParam.resize(InImage.size());

    mbf.MDetector.detect(InImage, mbf.Markers, mbf.CamParam,
                         mbf.MarkerSize); // detect markers computing R and T information
    // Detection of the board
    mbf.BoardDetector.detect(mbf.Markers, mbf.BoardConfig, mbf.Board, mbf.CamParam, mbf.MarkerSize);

    int mode = cv::FileStorage::Mode(generateResults);
    cv::FileStorage fs(TESTDATA_PATH "board/expected_gl.yml", mode);

    std::vector<cv::Vec<double, 16> > gldata(mbf.Markers.size() + 2), expected;

    mbf.CamParam.Distorsion.setTo(0); // silence cerr spam

    mbf.CamParam.glGetProjectionMatrix(InImage.size(), InImage.size(), gldata[0].val, 0.5, 10);
    mbf.Board.glGetModelViewMatrix(gldata[1].val);

    for (size_t i = 0; i < mbf.Markers.size(); i++) {
        mbf.Markers[i].glGetModelViewMatrix(gldata[i + 2].val);
    }

    if (generateResults) {
        fs << "gldata"
           << "[";
        for (size_t i = 0; i < gldata.size(); i++) {
            fs << gldata[i];
        }
        fs << "]";
        return;
    }

    // load results
    cv::FileNode gls = fs["gldata"];
    expected.resize(gls.size());

    for (size_t i = 0; i < expected.size(); i++) {
        gls[i] >> expected[i];
    }

    // now check the results
    for (size_t i = 0; i < expected.size(); i++) {
        for (int j = 0; j < 16; j++) {
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
    using namespace aruco;

    MarkerFixture mf;
    Dictionary dictionary;

    std::vector<Marker> expected;

    dictionary.fromFile(TESTDATA_PATH "hrm/dictionaries/d4x4_100.yml");
    HighlyReliableMarkers::loadDictionary(dictionary);

    cv::Mat frameImage = cv::imread(TESTDATA_PATH "hrm/image-test.png");

    mf.CamParam.readFromXMLFile(TESTDATA_PATH "hrm/intrinsics.yml");
    mf.CamParam.resize(frameImage.size());

    mf.MDetector.enableLockedCornersMethod(false);
    mf.MDetector.setMakerDetectorFunction(aruco::HighlyReliableMarkers::detect);
    mf.MDetector.setThresholdParams(21, 7);
    mf.MDetector.setCornerRefinementMethod(aruco::MarkerDetector::LINES);
    mf.MDetector.setWarpSize((dictionary[0].n() + 2) * 8);
    mf.MDetector.setMinMaxSize(0.005, 0.5);

    cv::FileStorage fs(TESTDATA_PATH "hrm/expected.yml", generateResults ? 1 : 0);

    mf.MDetector.detect(frameImage, mf.Markers, mf.CamParam, mf.MarkerSize);

    if (generateResults) {
        fs << "Markers" << mf.Markers;
        return;
    }

    fs["Markers"] >> expected;

    // now check the results
    ASSERT_EQ(expected.size(), mf.Markers.size());

    for (size_t i = 0; i < mf.Markers.size(); i++) {
        EXPECT_EQ(expected[i].id, mf.Markers[i].id);

        EXPECT_FLOAT_EQ(expected[i].getCenter().x, mf.Markers[i].getCenter().x);
        EXPECT_FLOAT_EQ(expected[i].getCenter().y, mf.Markers[i].getCenter().y);

        for (auto j = 0; j < 3; j++) {
            EXPECT_FLOAT_EQ(expected[i].Tvec(j), mf.Markers[i].Tvec(j));
            EXPECT_FLOAT_EQ(expected[i].Rvec(j), mf.Markers[i].Rvec(j));
        }
    }
}

TEST(Aruco, RefineFail) {
    using namespace aruco;

    MarkerFixture mf;
    Dictionary dictionary;

    dictionary.fromFile( TESTDATA_PATH "hrm/dictionaries/d4x4_100.yml" );
    HighlyReliableMarkers::loadDictionary( dictionary );

    cv::Mat frameImage = cv::imread( TESTDATA_PATH "hrm/refine-fail.png" );

    mf.CamParam.readFromXMLFile( TESTDATA_PATH "hrm/intrinsics.yml" );
    mf.CamParam.resize( frameImage.size() );

    mf.MDetector.enableLockedCornersMethod( false );
    mf.MDetector.setMakerDetectorFunction(HighlyReliableMarkers::detect );
    mf.MDetector.setThresholdParams( 21, 7 );
    mf.MDetector.setCornerRefinementMethod(MarkerDetector::LINES );
    mf.MDetector.setMinMaxSize( 0.005, 0.5 );
    mf.MDetector.setWarpSize( 48 );

    mf.MDetector.detect( frameImage, mf.Markers, mf.CamParam, mf.MarkerSize );
}
