#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include <aruco.h>

#define TESTADATA_PATH "../testdata/"

TEST(ArucoPerf, Single) {
    // TODO something like:
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
