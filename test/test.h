/*
 * test.h
 *
 *  Created on: 27.07.2015
 *      Author: parojtbe
 */

#pragma once

#include <aruco.h>

/// Structures needed to set up marker tracking
struct MarkerFixture {
    aruco::CameraParameters CamParam;
    aruco::MarkerDetector MDetector;
    std::vector<aruco::Marker> Markers;
    float MarkerSize = 1;
};

/// Structures needed to set up marker board tracking
struct MarkerBoardFixture : public MarkerFixture {
    aruco::BoardConfiguration BoardConfig;
    aruco::BoardDetector BoardDetector;
    aruco::Board Board;
};


